"""Persistent verified-replay learner, separate from completed-theorem experience.

One coordinator owns a learner run. Every committed step gets an immutable
checkpoint before another step or release is allowed. This first implementation
checkpoints every step; it never retries an uncertain training operation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading

from .identifiers import validate_identifier
from .learner_release_store import canonical_bytes, content_sha256
from .verified_objective import OBJECTIVE_KIND, integer


def _no_links(path):
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in (path, *path.parents)):
        raise ValueError("learner journal/lease paths must not contain links")


def _write(path, value):
    _no_links(path)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value)); stream.flush(); os.fsync(stream.fileno())
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _read_control(path):
    from cpu_runtime.verified_dataset_store import safe_directory
    with safe_directory(path.parent) as directory:
        raw = directory.read(path.name)
    value = json.loads(raw)
    if canonical_bytes(value) != raw:
        raise ValueError("learner control must be finite canonical JSON without duplicate keys")
    return value


def _sync_directory(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class _WriterLease:
    """OS lock, not a stale PID guess. Store owners must not unlink lease files."""
    def __init__(self, root, run_sha):
        directory = Path(root)/"writers"; _no_links(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/(run_sha+".lock"); _no_links(path)
        self.stream = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.stream.seek(0, 2) == 0:
                    self.stream.write(b"0"); self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            elif os.name == "posix":
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                raise ValueError("unsupported learner writer-lock platform")
        except BaseException:
            self.stream.close(); raise

    def close(self):
        if self.stream.closed:
            return
        if os.name == "nt":
            import msvcrt
            self.stream.seek(0); msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def describe_dataset(backend, pin):
    """Admission happens through the production full Lean evidence loader."""
    from cpu_runtime.verified_dataset_store import read_bundle
    dataset = backend._load_dataset(pin)
    bundle = read_bundle(backend.dataset_root/pin)
    if (hashlib.sha256(bundle["dataset.json"]).hexdigest() != pin
            or hashlib.sha256(bundle["session.json"]).hexdigest() != dataset["inputs_sha256"]["session.json"]):
        raise ValueError("dataset changed while binding its actor lineage")
    session = json.loads(bundle["session.json"])
    lineage = session.get("lineage") or {}
    return {"dataset_sha256": pin, "profile": dataset["profile"],
        "replay_receipt_sha256": dataset["replay_receipt_sha256"], "trace_sha256": dataset["trace_sha256"],
        "source_session_id": dataset["session_id"], "theorem_sha256": dataset["theorem_sha256"],
        "rows": len(dataset["rows"]), "source_model_release_sha256": lineage.get("model_release_sha256")}


def publish_checkpoint(runtime, *, checkpoint_sha256, journal_root, expected_run=None):
    """Publish a pinned checkpoint even when its original learner is absent.

    A new journal is required. An ambiguous publication is resolved by reading
    this intent and the content-addressed store, never by retrying a learn.
    """
    if runtime.learner_releases is None:
        raise ValueError("explicit learner store required")
    checkpoint = runtime.learner_releases.load_checkpoint(checkpoint_sha256)
    if expected_run is not None and checkpoint["run"] != expected_run:
        raise ValueError("checkpoint belongs to a different learner run")
    if runtime.actor.submit(runtime.backend.experience_contract) != checkpoint["run"]["contract"]:
        raise ValueError("publication backend contract differs from checkpoint")
    journal = Path(journal_root).absolute(); _no_links(journal)
    journal.mkdir(parents=True, exist_ok=False)
    _write(journal/"intent.json", {"checkpoint_sha256": checkpoint_sha256,
        "run_sha256": content_sha256(checkpoint["run"])})
    weights = runtime.actor.submit(lambda: runtime.backend.extract_checkpoint_weights(
        checkpoint["logical_state"], checkpoint["backend_state"]))
    release = runtime.learner_releases.publish(checkpoint_sha256, weights)
    metadata = runtime.learner_releases.load_release(release)[0]
    _write(journal/"published.json", metadata)
    return metadata


class LearnerCoordinator:
    BACKEND_KIND = "verified-replay"
    RUN_SCHEMA = "reap.learner.run.v1"
    DATA_SCHEMA = "reap.learner.data-receipt.v1"
    OBJECTIVE_KIND = OBJECTIVE_KIND

    def __init__(self, runtime, *, learner_id, dataset_pins, journal_root, batch_size,
                 scope=None, implementation=None, initial_model_release_sha256=None):
        self._configure(runtime, learner_id, journal_root)
        integer(batch_size, "learner batch_size", 1, 32)
        if not dataset_pins or len(set(dataset_pins)) != len(dataset_pins):
            raise ValueError("learner requires a unique explicit seed catalog")
        catalog = [describe_dataset(runtime.backend, pin) for pin in dataset_pins]
        contract = runtime.actor.submit(runtime.backend.experience_contract)
        if contract.get("backend") != "verified-replay":
            raise ValueError("persistent learner requires the verified replay profile")
        self.sampler_state = {"step": 0, "cursor": 0}
        self.run = {"schema_version": "reap.learner.run.v1", "role": "learner",
            "learner_id": learner_id, "backend_session_id": learner_id,
            "initialization": ({"kind": "base"} if initial_model_release_sha256 is None else
                {"kind": "learner_release", "release_sha256": initial_model_release_sha256}),
            "contract": contract, "seed": int.from_bytes(hashlib.sha256(learner_id.encode()).digest()[:8], "big") % 2**63,
            "scope": scope or {"kind": "generalist"}, "catalog": catalog,
            "sampler": {"kind": "cyclic_rows_v1", "batch_size": batch_size, "initial_state": self.sampler_state},
            "implementation": implementation or {"coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        self._start_run(catalog, initial_model_release_sha256)

    def _start_run(self, catalog, initial_model_release_sha256=None):
        """Shared durable creation; profile-specific admission precedes allocation."""
        runtime, learner_id = self.runtime, self.learner_id
        self.run_sha = self.store.create_run(self.run)
        self.lease = _WriterLease(self.store.root, self.run_sha)
        self.control = self.store.root/"control"/self.run_sha
        created_here = False
        try:
            _no_links(self.control)
            self.control.mkdir(parents=True, exist_ok=False)
            _sync_directory(self.control.parent)
            _sync_directory(self.store.root)
            _write(self.control/"created.json", {"run_sha256": self.run_sha})
            self.journal.mkdir(parents=True, exist_ok=False)
            _write(self.journal/"create-intent.json", {"run_sha256": self.run_sha, "run": self.run})
            created = runtime.create_session(learner_id, role="learner", model_release_sha256=initial_model_release_sha256)
            created_here = True
            if created.get("role") != "learner" or created["theorem_id"] is not None or created["policy_version"] != 0:
                raise ValueError("learner initialization identity mismatch")
            _write(self.journal/"created.json", created)
        except BaseException:
            self.blocked = True
            try:
                if created_here:
                    runtime.delete_session(learner_id)
            finally:
                self.lease.close()
            raise
        self.catalog, self.parent_checkpoint, self.parent_receipt = catalog, None, None

    def _configure(self, runtime, learner_id, journal_root):
        validate_identifier(learner_id, kind="learner_id")
        if runtime.learner_releases is None:
            raise ValueError("learner requires an explicit learner release store")
        if (runtime.actor.submit(runtime.backend.experience_contract).get("backend") != self.BACKEND_KIND
                or runtime.learner_releases.RUN_SCHEMA != self.RUN_SCHEMA):
            raise ValueError("learner coordinator/backend/store profile mismatch")
        self.runtime, self.store, self.learner_id = runtime, runtime.learner_releases, learner_id
        self.journal = Path(journal_root).absolute(); _no_links(self.journal)
        self.blocked, self.closed, self.lock = False, False, threading.RLock()
        self._publication_head_cache = None

    def _check_live(self, *, allow_unpublished=False):
        if self.blocked or self.closed:
            raise RuntimeError("learner is blocked/closed; inspect its durable intent, never retry unknown work")
        state = self.runtime.sessions.get(self.learner_id)
        if state.role != "learner" or state.theorem_id is not None or state.completed or state.policy_version != self.sampler_state["step"]:
            raise ValueError("live learner differs from its last durable step")
        if self._durable_head() != self.parent_checkpoint:
            raise ValueError("live learner differs from durable run head")
        from .release_head import publication_state
        publication = publication_state(self.store, self.run_sha, verify_release=False)
        if publication is not None:
            if canonical_bytes(publication["head"]) != canonical_bytes(self._publication_head_cache):
                raise RuntimeError("publication head differs from owning coordinator validation")
            if publication["pending"] is not None and not allow_unpublished:
                raise RuntimeError("unpublished/unknown commit blocks new training")

    def _durable_head(self):
        """Every mutation intent survives coordinator/process reconstruction.

        Missing commit means unknown, including checkpoint-written/ack-lost.
        No automatic retry, rollback to an older head, or guessed reconciliation.
        """
        from .release_head import read_training_commits, publication_state
        commits = read_training_commits(self.control, self.run_sha)
        state = publication_state(self.store, self.run_sha, commits=commits, verify_release=False)
        if state is not None and state["pending"] is not None and state["pending"]["intent"] is not None:
            raise RuntimeError("unresolved publication intent; reconcile before resuming")
        return commits[-1]["checkpoint_sha256"] if commits else None

    def enable_release_head(self):
        from .release_head import enable
        return enable(self)

    def train_next_and_publish(self, *, append_dataset_pins=()):
        with self.lock:
            from .release_head import publication_state, read_publication_head
            if publication_state(self.store, self.run_sha, verify_release=False) is None:
                raise RuntimeError("explicit release-head enable required")
            committed = self.train_next(append_dataset_pins=append_dataset_pins)
            release = self.publish()
            return {**committed, "release": release,
                    "head_receipt": read_publication_head(self.store, self.run_sha)}

    def _prepare_next(self, append_dataset_pins):
        catalog = deepcopy(self.catalog)
        existing = {d["dataset_sha256"] for d in catalog}
        for pin in append_dataset_pins:
            if pin in existing:
                raise ValueError("catalog append cannot duplicate or replace a dataset")
            catalog.append(describe_dataset(self.runtime.backend, pin)); existing.add(pin)
        rows = [{"dataset_sha256": d["dataset_sha256"], "row": i} for d in catalog for i in range(d["rows"])]
        cursor = self.sampler_state["cursor"]
        batch = [rows[(cursor+i) % len(rows)] for i in range(self.run["sampler"]["batch_size"])]
        return catalog, batch, {"step": self.sampler_state["step"]+1, "cursor": cursor+len(batch)}

    def _data_extras(self):
        return {}

    def train_next(self, *, append_dataset_pins=()):
        with self.lock:
            self._check_live()
            catalog, batch, after_sampler = self._prepare_next(append_dataset_pins)
            step = after_sampler["step"]
            event = {"kind": self.OBJECTIVE_KIND, "event_id": f"lr-{self.run_sha[:16]}-{step}",
                "session_id": self.learner_id, "policy_version": step-1, "samples": batch}
            directory = self.journal/f"step-{step:08d}"
            try:
                intent = {"run_sha256": self.run_sha, "step": step, "event": event,
                    "sampler_before": self.sampler_state, "catalog": catalog, "parent_checkpoint_sha256": self.parent_checkpoint}
                _write(self.control/f"intent-{step:08d}.json", intent)
                directory.mkdir(exist_ok=False)
                _write(directory/"intent.json", intent)
                receipt = self.runtime.learn(self.learner_id, expected_policy_version=step-1, event=event)
                _write(directory/"runtime-receipt.json", receipt)
                data = {"schema_version": self.DATA_SCHEMA, "run_sha256": self.run_sha,
                    "step": step, "parent_receipt_sha256": self.parent_receipt,
                    "event": event, "event_sha256": content_sha256(event), "runtime_receipt": receipt,
                    "runtime_receipt_sha256": content_sha256(receipt), "batch_refs": batch,
                    "sampler_before_sha256": content_sha256(self.sampler_state),
                    "sampler_after_sha256": content_sha256(after_sampler),
                    "catalog": catalog, "catalog_sha256": content_sha256(catalog), **self._data_extras()}
                _write(directory/"data-receipt.json", data)
                path = self.runtime.snapshot(self.learner_id, f"learner-step-{step:08d}")
                logical, backend = self.runtime.snapshots.load(self.learner_id, path.name)
                pin = self.store.create_checkpoint(self.run_sha, logical, backend, data, after_sampler,
                                                  parent_checkpoint_sha256=self.parent_checkpoint)
                # A missing journal acknowledgment never authorizes another step.
                _write(directory/"checkpoint.json", {"checkpoint_sha256": pin, "step": step})
                _write(self.control/f"commit-{step:08d}.json", {"step": step,
                    "checkpoint_sha256": pin, "intent_sha256": content_sha256(intent)})
            except BaseException:
                self.blocked = True
                raise
            self.parent_checkpoint, self.parent_receipt = pin, content_sha256(data)
            self.sampler_state, self.catalog = after_sampler, catalog
            return {"checkpoint_sha256": pin, "step": step, "runtime_receipt": receipt}

    def publish(self, checkpoint_sha256=None):
        with self.lock:
            self._check_live(allow_unpublished=True)
            from .release_head import publication_state, publish as publish_head
            if publication_state(self.store, self.run_sha, verify_release=False) is not None:
                try:
                    return publish_head(self, checkpoint_sha256)
                except BaseException:
                    self.blocked = True
                    raise
            pin = checkpoint_sha256 or self.parent_checkpoint
            if pin is None:
                raise ValueError("learner has no completed training checkpoint")
            try:
                return publish_checkpoint(self.runtime, checkpoint_sha256=pin,
                    journal_root=self.journal/("publish-"+pin), expected_run=self.run)
            except BaseException:
                self.blocked = True
                raise

    @classmethod
    def restore(cls, runtime, *, checkpoint_sha256, journal_root):
        if runtime.learner_releases is None:
            raise ValueError("explicit learner store required")
        checkpoint = runtime.learner_releases.load_checkpoint(checkpoint_sha256)
        run = checkpoint["run"]
        if runtime.actor.submit(runtime.backend.experience_contract) != run["contract"]:
            raise ValueError("learner restore contract differs from checkpoint")
        self = cls.__new__(cls); self._configure(runtime, run["learner_id"], journal_root)
        self.run, self.run_sha = run, content_sha256(run)
        self.lease = _WriterLease(self.store.root, self.run_sha)
        self.control = self.store.root/"control"/self.run_sha
        created_here = False
        try:
            if self._durable_head() != checkpoint_sha256:
                raise ValueError("restore must use durable learner head; historical checkpoints are publication-only")
            from .release_head import publication_state
            publication = publication_state(self.store, self.run_sha)
            self._publication_head_cache = None if publication is None else publication["head"]
            self.journal.mkdir(parents=True, exist_ok=False)
            _write(self.journal/"restore-intent.json", {"checkpoint_sha256": checkpoint_sha256, "run_sha256": self.run_sha})
            initialization = run["initialization"]
            runtime.create_session(self.learner_id, role="learner",
                model_release_sha256=initialization.get("release_sha256"))
            created_here = True
            # Full content pin fits the snapshot identifier limit. A second
            # restore of this head only reads and verifies the existing bytes.
            restore_name = checkpoint_sha256
            existing = runtime.snapshots.root/self.learner_id/restore_name
            if existing.exists() or existing.is_symlink():
                logical, backend = runtime.snapshots.load(self.learner_id, restore_name)
                if logical != checkpoint["logical_state"] or backend != checkpoint["backend_state"]:
                    raise ValueError("existing restore snapshot differs from learner checkpoint")
            else:
                runtime.snapshots.create(self.learner_id, restore_name, session_state=checkpoint["logical_state"],
                                         backend_state=checkpoint["backend_state"])
            restored = runtime.restore(self.learner_id, restore_name)
            if restored != checkpoint["logical_state"]:
                raise ValueError("learner restore logical state mismatch")
            _write(self.journal/"restored.json", restored)
        except BaseException:
            self.blocked = True
            try:
                if created_here:
                    runtime.delete_session(self.learner_id)
            finally:
                self.lease.close()
            raise
        self.parent_checkpoint = checkpoint_sha256
        self.parent_receipt = content_sha256(checkpoint["data_receipt"])
        self.sampler_state = checkpoint["sampler_state"]
        self.catalog = checkpoint["data_receipt"]["catalog"]
        return self

    def close(self):
        """Release coordinator ownership only; caller explicitly manages GPU state."""
        with self.lock:
            if not self.closed:
                self.closed = True
                if hasattr(self, "lease"):
                    self.lease.close()
