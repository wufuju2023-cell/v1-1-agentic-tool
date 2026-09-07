"""High-level session, OpenAI Chat, learn, and snapshot orchestration."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from .actor import GpuActor
from .backend import Backend
from .errors import DuplicateSessionError, EventConflictError, SessionNotFoundError, VersionConflictError
from .experience_store import ExperienceStore, require_sha256
from .identifiers import contained_path, validate_identifier
from .schemas import chat_response, parse_chat_request, value_chat_response
from .session_store import SessionState, SessionStore
from .snapshot_store import SnapshotStore


def _event_digest(event: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("learn event must be finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _write_retirement_record(root: Path, session_id: str, name: str, record: dict[str, Any]) -> None:
    """Publish one immutable, fsynced JSON record without replacing evidence."""
    directory = contained_path(root, session_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = contained_path(directory, name)
    temporary = contained_path(directory, ".tmp-" + uuid.uuid4().hex)
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-overwrite publication, unlike os.replace of a receipt.
        os.link(temporary, target)
        _sync_directory(directory)
        _sync_directory(root)
        _sync_directory(root.parent)
    finally:
        temporary.unlink(missing_ok=True)


class GpuRuntime:
    def __init__(
        self,
        *,
        backend: Backend,
        snapshot_root: Path,
        actor: GpuActor | None = None,
        sessions: SessionStore | None = None,
        experience_root: Path | None = None,
        learner_release_root: Path | None = None,
        learner_profile: str | None = None,
        max_resident_sessions: int | None = None,
    ) -> None:
        if max_resident_sessions is not None and (type(max_resident_sessions) is not int or max_resident_sessions < 1):
            raise ValueError("max_resident_sessions must be a positive integer or None")
        if learner_profile is not None and (learner_profile != "continual-mixed-v3" or learner_release_root is None):
            raise ValueError("explicit continual-mixed-v3 profile requires a learner release root")
        if learner_profile is not None and backend.experience_contract().get("backend") != "mixed-replay":
            raise ValueError("continual-mixed-v3 requires the exact mixed backend")
        self.backend = backend
        self.actor = actor or GpuActor()
        self.sessions = sessions or SessionStore()
        self.snapshots = SnapshotStore(snapshot_root)
        self.experiences = ExperienceStore(experience_root or snapshot_root / "_experiences")
        self.learner_releases = None
        if learner_release_root is not None:
            from .learner_release_store import LearnerReleaseStore
            if self.actor.submit(backend.experience_contract).get("backend") == "mixed-replay":
                if learner_profile == "continual-mixed-v3":
                    from .continual_mixed_store import ContinualMixedStore
                    self.learner_releases = ContinualMixedStore(learner_release_root)
                else:
                    from .mixed_learner_store import MixedLearnerStore
                    self.learner_releases = MixedLearnerStore(learner_release_root)
            else:
                self.learner_releases = LearnerReleaseStore(learner_release_root)
        self._unusable_sessions: set[str] = set()
        self.max_resident_sessions = max_resident_sessions
        # Allocation ownership belongs to this runtime. Quarantine residues also
        # consume slots even when logical SessionStore creation never completed.
        # A supplied SessionStore can already contain allocations. SessionStore
        # has no listing API; take its identity set under its own guard once.
        with self.sessions._guard:
            self._resident_session_ids: set[str] = set(self.sessions._states)
        self._retired_session_ids: set[str] = set()
        self._retirement_root = contained_path(self.snapshots.root, "_retirements")
        # This cache is deliberately process-local and never restored from disk.
        # Direct backend calls outside this runtime are unsupported for reuse.
        self._reuse_revision = 0
        self._session_revisions: dict[str, int] = {}
        self._snapshot_witnesses: dict[str, dict[str, Any]] = {}

    def _invalidate_snapshot_reuse(self, session_id: str) -> None:
        """Call before any backend operation, including failed/pure-looking ones."""
        self._snapshot_witnesses.pop(session_id, None)
        if session_id in self._resident_session_ids:
            self._reuse_revision += 1
            self._session_revisions[session_id] = self._reuse_revision

    def _ensure_usable(self, session_id: str) -> None:
        if session_id in self._unusable_sessions:
            raise RuntimeError(
                f"session {session_id} is quarantined after failed rollback or initialization cleanup; operator inspection required"
            )

    def _rollback(self, session_id: str, backend_state: dict[str, Any], session_state: dict[str, Any]) -> None:
        try:
            self.backend.import_session(session_id, backend_state)
            self.sessions.replace_locked(session_id, SessionState.from_snapshot(session_state))
        except BaseException as exc:
            # The policy version cannot describe partially restored parameters.
            # Keep other sessions available, but never serve this one again.
            self._unusable_sessions.add(session_id)
            raise RuntimeError(f"rollback failed for session {session_id}; session quarantined") from exc

    def create_session(self, session_id: str, *, theorem_id: str | None = None,
                       experience_id: str | None = None, experience_weights_sha256: str | None = None,
                       experience_snapshot_sha256: str | None = None,
                       model_release_sha256: str | None = None, role: str | None = None) -> dict[str, Any]:
        session_id = validate_identifier(session_id, kind="session_id")
        role = ("actor" if model_release_sha256 is not None else "theorem") if role is None else role
        if not isinstance(role, str) or role not in {"theorem", "learner", "actor"}:
            raise ValueError("unknown session role")
        if model_release_sha256 is not None:
            require_sha256(model_release_sha256, "learner model release")
            if (experience_id is not None or experience_weights_sha256 is not None
                    or experience_snapshot_sha256 is not None or role == "theorem"):
                raise ValueError("learner release and legacy experience initialization are mutually exclusive")
            if self.learner_releases is None:
                raise ValueError("explicit learner release store is not configured")
        if role == "learner" and (theorem_id is not None or experience_id is not None):
            raise ValueError("learner has no theorem or completed-proof experience identity")
        if role == "actor":
            if model_release_sha256 is None:
                raise ValueError("actor requires an explicit learner model release")
            require_sha256(theorem_id, "actor theorem SHA256")
        if theorem_id is not None:
            theorem_id = validate_identifier(theorem_id, kind="theorem_id")
        if experience_id is not None:
            experience_id = validate_identifier(experience_id, kind="experience_id")
            if theorem_id is None:
                raise ValueError("experience initialization requires a new theorem_id")
        pinned = experience_weights_sha256 is not None or experience_snapshot_sha256 is not None
        if pinned:
            if experience_id is None:
                raise ValueError("experience content pins require experience_id")
            require_sha256(experience_weights_sha256, "experience weights pin")
            require_sha256(experience_snapshot_sha256, "experience snapshot pin")

        def operation() -> dict[str, Any]:
            if session_id in self._retired_session_ids:
                raise ValueError("retired session_id cannot be reused in this runtime")
            self._ensure_usable(session_id)
            allocated = self._resident_session_ids | self._unusable_sessions
            if (self.max_resident_sessions is not None and session_id not in allocated
                    and len(allocated) >= self.max_resident_sessions):
                raise ValueError("resident session capacity reached; wait for an explicitly retired session")
            release = weights = None
            if role != "theorem" and self.backend.experience_contract().get("backend") not in {"verified-replay", "mixed-replay"}:
                raise ValueError("learner/actor roles require verified-replay or mixed-replay backend")
            if model_release_sha256 is not None:
                release, weights = self.learner_releases.load_release(model_release_sha256)
                if (release["model_release_sha256"] != model_release_sha256
                        or release["source"]["learner_id"] == session_id):
                    raise ValueError("release identity conflicts with new learner/actor")
                if weights["contract"] != self.backend.experience_contract():
                    raise ValueError("learner release base/model/objective compatibility mismatch")
            if experience_id is not None:
                release, weights = self.experiences.load(experience_id)
                if pinned and (release["weights_sha256"] != experience_weights_sha256
                               or release["source"]["snapshot_sha256"] != experience_snapshot_sha256):
                    raise ValueError("experience content differs from pinned release; no session allocated")
                if (release["source"]["session_id"] == session_id
                        or release["source"]["theorem_id"] == theorem_id):
                    raise ValueError("cross-theorem initialization requires distinct session and theorem identities")
                if weights["contract"] != self.backend.experience_contract():
                    raise ValueError("experience base/model/objective compatibility mismatch")
            if session_id in self._resident_session_ids:
                raise DuplicateSessionError(f"session already exists: {session_id}")
            # Reserve before calling the backend: even create itself can fail
            # after allocating an adapter, before logical SessionStore creation.
            self._resident_session_ids.add(session_id)
            self._invalidate_snapshot_reuse(session_id)
            try:
                try:
                    metadata = self.backend.create_session(session_id)
                except DuplicateSessionError:
                    # An unexpected existing backend allocation is not ours to
                    # delete. Keep its capacity reserved for inspection.
                    self._unusable_sessions.add(session_id)
                    raise
                if weights is not None:
                    metadata.update(self.backend.initialize_from_experience(session_id, weights))
                    metadata["lineage"] = {("model_release_sha256" if model_release_sha256 is not None else "experience_id"):
                        model_release_sha256 if model_release_sha256 is not None else experience_id,
                        "weights_sha256": release["weights_sha256"], "source": release["source"],
                        "reset": ["optimizer", "rng", "buffer", "policy_version", "event_receipts"]}
                metadata["role"] = role
                metadata["theorem_id"] = theorem_id
                state = self.sessions.create(session_id, metadata)
            except BaseException:
                if session_id in self._unusable_sessions:
                    raise
                try:
                    self.backend.delete_session(session_id)
                    self._resident_session_ids.discard(session_id)
                    self._session_revisions.pop(session_id, None)
                except BaseException as exc:
                    self._unusable_sessions.add(session_id)
                    raise RuntimeError("failed initialization cleanup; session quarantined") from exc
                raise
            return state.snapshot()

        return self.actor.submit(operation)

    def delete_session(self, session_id: str) -> None:
        session_id = validate_identifier(session_id, kind="session_id")

        def operation() -> None:
            with self.sessions.locked(session_id):
                self._invalidate_snapshot_reuse(session_id)
                self.backend.delete_session(session_id)
                self.sessions.delete(session_id)
                self._unusable_sessions.discard(session_id)
                self._resident_session_ids.discard(session_id)
                self._session_revisions.pop(session_id, None)

        self.actor.submit(operation)

    def retire_session(self, session_id: str, name: str, *, expected_policy_version: int,
                       reuse_snapshot: bool = False) -> dict[str, Any]:
        """Save a final snapshot, then release this session's resident state.

        Caller must have finished its Lean process and resolved every request.
        This operation does not infer completion from a successful proof, and
        retirement does not attest that an experience is accepted. A repeated
        call is rejected; recover a lost ACK by reading the immutable receipt.
        Tombstones cover this runtime only; restart uses a new run root.
        """
        session_id = validate_identifier(session_id, kind="session_id")
        name = validate_identifier(name, kind="snapshot name")
        if type(expected_policy_version) is not int or expected_policy_version < 0:
            raise ValueError("expected_policy_version must be a non-negative integer")
        if type(reuse_snapshot) is not bool:
            raise ValueError("reuse_snapshot must be explicitly boolean")

        def operation() -> dict[str, Any]:
            if session_id in self._retired_session_ids:
                raise ValueError("retirement already attempted; inspect its receipt, do not resubmit")
            with self.sessions.locked(session_id) as state:
                self._ensure_usable(session_id)
                if state.policy_version != expected_policy_version:
                    raise VersionConflictError("retirement policy_version mismatch")
                if state.buffer_metadata.get("pending_event_ids"):
                    raise ValueError("cannot retire with pending learning events")
                reuse_fields = {}
                if reuse_snapshot:
                    witness = self._snapshot_witnesses.get(session_id)
                    if (witness is None or witness["name"] != name
                            or witness["revision"] != self._session_revisions.get(session_id)):
                        raise ValueError("snapshot reuse requires an unchanged same-runtime witness; refusing to re-snapshot")
                    logical = self.snapshots.verify_for_reuse(session_id, name,
                        expected_manifest_sha256=witness["manifest_sha256"])
                    # Canonical finite JSON comparison also distinguishes
                    # bool/int/float values that Python dict equality equates.
                    if (_event_digest(logical) != _event_digest(witness["logical"])
                            or _event_digest(logical) != _event_digest(state.snapshot())):
                        raise ValueError("snapshot reuse logical state differs from live session")
                    path = contained_path(self.snapshots.root, session_id, name)
                    reuse_fields = {"snapshot_mode": "reuse_verified", "snapshot_revision": witness["revision"]}
                    snapshot_sha256 = witness["manifest_sha256"]
                    self._invalidate_snapshot_reuse(session_id)
                else:
                    self._invalidate_snapshot_reuse(session_id)
                    path = self.snapshots.create(session_id, name, session_state=state.snapshot(),
                                                 backend_state=self.backend.export_session(session_id))
                    snapshot_sha256 = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                _sync_directory(path)
                _sync_directory(path.parent)
                _sync_directory(self.snapshots.root)
                receipt = {"schema_version": "reap.gpu.retirement.v2" if reuse_snapshot else "reap.gpu.retirement.v1", "session_id": session_id,
                    "policy_version": state.policy_version, "snapshot": name,
                    "snapshot_sha256": snapshot_sha256,
                    "status": "prepared", "mutation_retry_allowed": False,
                    "tombstone_scope": "current_runtime", **reuse_fields}
                # From this point a crash or exception is an unknown retirement,
                # never a reason to recreate this identity or reissue a delete.
                self._retired_session_ids.add(session_id)
                try:
                    _write_retirement_record(self._retirement_root, session_id, "prepared.json", receipt)
                    self.backend.delete_session(session_id)
                    self.sessions.delete(session_id)
                    self._resident_session_ids.discard(session_id)
                    self._session_revisions.pop(session_id, None)
                    released = {**receipt, "status": "released"}
                    _write_retirement_record(self._retirement_root, session_id, "released.json", released)
                except BaseException as exc:
                    self._unusable_sessions.add(session_id)
                    raise RuntimeError("retirement outcome incomplete; session quarantined; inspect saved records") from exc
                return released

        return self.actor.submit(operation)

    def retirement_receipt(self, session_id: str) -> dict[str, Any]:
        """Read one immutable retirement record without queueing GPU work.

        Absence is not proof that a request never executed. Only a validated
        released record confirms release; prepared requires reconciliation.
        """
        session_id = validate_identifier(session_id, kind="session_id")
        directory = contained_path(self._retirement_root, session_id)

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate retirement JSON key")
                result[key] = value
            return result

        def read_small_json(path, *, with_bytes=False):
            with path.open("rb") as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                raise ValueError("retirement metadata exceeds 64 KiB")
            value = json.loads(raw, object_pairs_hook=unique_keys)
            return (value, raw) if with_bytes else value

        records = {}
        fields = {"schema_version", "session_id", "policy_version", "snapshot", "snapshot_sha256",
                  "status", "mutation_retry_allowed", "tombstone_scope"}
        for status in ("prepared", "released"):
            try:
                record = read_small_json(contained_path(directory, status + ".json"))
            except FileNotFoundError:
                continue
            reused = isinstance(record, dict) and record.get("schema_version") == "reap.gpu.retirement.v2"
            expected_fields = fields | {"snapshot_mode", "snapshot_revision"} if reused else fields
            if (not isinstance(record, dict) or set(record) != expected_fields
                    or record["schema_version"] not in {"reap.gpu.retirement.v1", "reap.gpu.retirement.v2"}
                    or record["session_id"] != session_id or record["status"] != status
                    or type(record["policy_version"]) is not int or record["policy_version"] < 0
                    or record["mutation_retry_allowed"] is not False
                    or record["tombstone_scope"] != "current_runtime"):
                raise ValueError("invalid retirement receipt identity/schema")
            if reused and (record["snapshot_mode"] != "reuse_verified"
                           or type(record["snapshot_revision"]) is not int or record["snapshot_revision"] < 1):
                raise ValueError("invalid reused retirement snapshot mode/revision")
            validate_identifier(record["snapshot"], kind="snapshot name")
            digest = record["snapshot_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid retirement snapshot hash")
            records[status] = record
        if not records:
            raise SessionNotFoundError("no retirement receipt; absence does not permit mutation retry")
        if "prepared" not in records:
            raise ValueError("released retirement receipt is missing its prepared record")
        prepared = records["prepared"]
        if "released" in records and records["released"] != {**prepared, "status": "released"}:
            raise ValueError("prepared/released retirement identity mismatch")
        manifest_path = contained_path(self.snapshots.root, session_id, prepared["snapshot"], "manifest.json")
        try:
            manifest, manifest_bytes = read_small_json(manifest_path, with_bytes=True)
        except FileNotFoundError as exc:
            raise ValueError("retirement snapshot manifest missing") from exc
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != "reap.gpu.snapshot.v1"
                or manifest.get("session_id") != session_id or manifest.get("snapshot") != prepared["snapshot"]
                or hashlib.sha256(manifest_bytes).hexdigest() != prepared["snapshot_sha256"]):
            raise ValueError("retirement snapshot manifest identity/hash mismatch")
        return records.get("released", prepared)

    def policy(self, session_id: str, raw_request: dict[str, Any]) -> dict[str, Any]:
        request = parse_chat_request(raw_request)

        def operation() -> dict[str, Any]:
            with self.sessions.locked(session_id) as state:
                self._ensure_usable(session_id)
                self._invalidate_snapshot_reuse(session_id)
                contents, logprobs = self.backend.policy(session_id, request)
                if len(contents) != request.n or len(logprobs) != request.n:
                    raise RuntimeError("backend returned the wrong number of policy choices")
                return chat_response(
                    model=request.model,
                    contents=contents,
                    token_logprobs=logprobs if request.logprobs else None,
                    policy_version=state.policy_version,
                )

        return self.actor.submit(operation)

    def value(self, session_id: str, raw_request: dict[str, Any]) -> dict[str, Any]:
        request = parse_chat_request(raw_request)

        def operation() -> dict[str, Any]:
            with self.sessions.locked(session_id) as state:
                self._ensure_usable(session_id)
                self._invalidate_snapshot_reuse(session_id)
                score = self.backend.value(session_id, request)
                return value_chat_response(model=request.model, score=score, policy_version=state.policy_version)

        return self.actor.submit(operation)

    def learn(self, session_id: str, *, expected_policy_version: int, event: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(expected_policy_version, int) or isinstance(expected_policy_version, bool) or expected_policy_version < 0:
            raise ValueError("expected_policy_version must be a non-negative integer")
        if not isinstance(event, dict):
            raise ValueError("event must be an object")
        event_id = validate_identifier(event.get("event_id"), kind="event_id")
        digest = _event_digest(event)

        def operation() -> dict[str, Any]:
            with self.sessions.locked(session_id) as state:
                self._ensure_usable(session_id)
                self._invalidate_snapshot_reuse(session_id)
                receipt = state.event_receipts.get(event_id)
                if state.completed:
                    raise ValueError("completed experience source cannot learn")
                if state.role == "actor":
                    raise ValueError("fixed-release actor cannot perform local learning")
                if event.get('kind') == 'verified_success_discounted_v1':
                    if state.role != 'theorem' or not state.theorem_id or event.get('theorem_id') != state.theorem_id:
                        raise ValueError('successful finalization requires the exact theorem session identity')
                if receipt is not None:
                    if receipt.get("digest") != digest:
                        raise EventConflictError(f"event_id reused with different payload: {event_id}")
                    response = deepcopy(receipt["response"])
                    response["applied"] = False
                    response["idempotent"] = True
                    return response
                if state.policy_version != expected_policy_version:
                    raise VersionConflictError(
                        f"stale policy_version for {session_id}: expected {expected_policy_version}, current {state.policy_version}"
                    )
                old_session = state.snapshot()
                # Backend snapshots contain only this session's mutable state,
                # not the shared frozen base. Snapshot before any mutation.
                old_backend = self.backend.export_session(session_id)
                try:
                    state.buffer_metadata.setdefault("events", {})[event_id] = {
                        "digest": digest,
                        "status": "pending",
                        "policy_version": expected_policy_version,
                    }
                    state.buffer_metadata.setdefault("pending_event_ids", []).append(event_id)
                    result = self.backend.learn(session_id, event)
                    # Reject invalid/NaN diagnostics before committing a version
                    # or receipt, including a backend that mutated then returned
                    # an invalid result instead of raising an exception.
                    json.dumps({
                        "adapter": result.adapter_metadata,
                        "value": result.value_metadata,
                        "optimizer": result.optimizer_metadata,
                        "detail": result.detail,
                    }, allow_nan=False)
                    state.policy_version += 1
                    state.adapter_metadata = deepcopy(result.adapter_metadata)
                    state.value_metadata = deepcopy(result.value_metadata)
                    state.optimizer_metadata = deepcopy(result.optimizer_metadata)
                    state.buffer_metadata["events"][event_id]["status"] = "consumed"
                    state.buffer_metadata["pending_event_ids"].remove(event_id)
                    state.buffer_metadata.setdefault("consumed_event_ids", []).append(event_id)
                    response = {
                        "event_id": event_id,
                        "applied": True,
                        "idempotent": False,
                        "policy_version": state.policy_version,
                        "detail": deepcopy(result.detail),
                    }
                    state.event_receipts[event_id] = {"digest": digest, "response": deepcopy(response)}
                except BaseException:
                    self._rollback(session_id, old_backend, old_session)
                    raise
                return response

        return self.actor.submit(operation)

    def snapshot(self, session_id: str, name: str, *, for_experience: bool = False) -> Path:
        name = validate_identifier(name, kind="snapshot name")
        if type(for_experience) is not bool:
            raise ValueError("for_experience must be boolean")

        def operation() -> Path:
            with self.sessions.locked(session_id) as state:
                self._ensure_usable(session_id)
                self._invalidate_snapshot_reuse(session_id)
                logical = state.snapshot()
                backend_state = self.backend.export_session(session_id)
                if for_experience:
                    if state.role != "theorem":
                        raise ValueError("learner/actor cannot publish as a completed theorem")
                    if not state.theorem_id or state.policy_version < 1 or state.buffer_metadata.get("pending_event_ids"):
                        raise ValueError("experience candidate requires theorem identity and completed updates")
                    backend_state["experience_contract"] = self.backend.experience_contract()
                    logical["completed"] = True
                path = self.snapshots.create(
                    session_id,
                    name,
                    session_state=logical,
                    backend_state=backend_state,
                )
                if for_experience:
                    state.completed = True
                self._snapshot_witnesses[session_id] = {"name": name,
                    "revision": self._session_revisions[session_id], "logical": deepcopy(logical),
                    "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()}
                return path

        return self.actor.submit(operation)

    def publish_experience(self, session_id: str, name: str, experience_id: str,
                           acceptance: dict[str, Any]) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            self._invalidate_snapshot_reuse(session_id)
            raw, backend = self.snapshots.load(session_id, name)
            state = SessionState.from_snapshot(raw)
            if (state.session_id != session_id or not state.completed or not state.theorem_id
                    or state.policy_version < 1 or state.buffer_metadata.get("pending_event_ids")):
                raise ValueError("source must be an explicitly completed experience candidate")
            manifest = self.snapshots.root / session_id / name / "manifest.json"
            source = {"session_id": session_id, "theorem_id": state.theorem_id,
                      "policy_version": state.policy_version, "snapshot": name,
                      "snapshot_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "parent_experience_id": state.lineage.get("experience_id")}
            weights = self.backend.experience_weights(session_id, backend)
            return self.experiences.publish(experience_id, source=source, acceptance=acceptance, weights=weights)
        return self.actor.submit(operation)

    def restore(self, session_id: str, name: str) -> dict[str, Any]:
        name = validate_identifier(name, kind="snapshot name")
        raw_session, raw_backend = self.snapshots.load(session_id, name)
        restored = SessionState.from_snapshot(raw_session)
        if restored.session_id != session_id:
            raise ValueError("snapshot session id mismatch")

        def operation() -> dict[str, Any]:
            with self.sessions.locked(session_id) as current:
                self._ensure_usable(session_id)
                if current.role != restored.role:
                    raise ValueError("same-session restore cannot change learner/actor/theorem role")
                self._invalidate_snapshot_reuse(session_id)
                old_session = current.snapshot()
                if current.completed:
                    raise ValueError("completed experience source cannot restore")
                if current.theorem_id != restored.theorem_id or current.lineage != restored.lineage:
                    raise ValueError("snapshot theorem/initialization lineage mismatch")
                old_backend = self.backend.export_session(session_id)
                try:
                    self.backend.import_session(session_id, raw_backend)
                    self.sessions.replace_locked(session_id, restored)
                except BaseException:
                    self._rollback(session_id, old_backend, old_session)
                    raise
                return restored.snapshot()

        return self.actor.submit(operation)

    def inspect_backend(self, session_id: str) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            self._ensure_usable(session_id)
            self._invalidate_snapshot_reuse(session_id)
            return self.backend.export_session(session_id)

        return self.actor.submit(operation)

    def close(self) -> None:
        self.actor.close()

    def __enter__(self) -> "GpuRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
