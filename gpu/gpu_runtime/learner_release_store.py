"""Immutable learner runs/checkpoints/parameter releases, separate from theorem experience.

The operator owns this local registry. Hashes are integrity pins, not signatures.
The Coordinator validates dataset evidence and the backend extracts/checks tensor
weights. This stdlib store checks their strict JSON envelopes and receipt chain;
it never pretends to decode or validate opaque torch tensor payloads itself.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import errno
import hashlib
import json
import os
from pathlib import Path
from typing import Any, TypedDict
import uuid

from cpu_runtime.verified_dataset_store import safe_directory, _publish_noreplace
from .identifiers import validate_identifier
from .experience_store import require_sha256
from .verified_objective import DATA_PROFILE, OBJECTIVE_KIND


class LearnerStoreError(ValueError):
    pass


class CheckpointPackage(TypedDict):
    manifest: dict
    run: dict
    logical_state: dict
    backend_state: dict
    data_receipt: dict
    sampler_state: dict


def require(ok: bool, message: str) -> None:
    if not ok:
        raise LearnerStoreError(message)


def canonical_bytes(value: Any) -> bytes:
    """Finite, string-keyed canonical JSON, without a trailing newline."""
    def check(item):
        if isinstance(item, dict):
            require(all(isinstance(k, str) for k in item), "JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        else:
            require(item is None or type(item) in (str, int, float, bool), "ordinary JSON values required")
    check(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise LearnerStoreError("finite canonical JSON required") from exc


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _decode(raw: bytes) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=unique)
        require(canonical_bytes(result) == raw, "stored JSON must be canonical and finite")
        return result
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LearnerStoreError("invalid stored JSON") from exc


def _keys(value, fields, label):
    require(isinstance(value, dict) and set(value) == set(fields), f"{label} fields mismatch")


def _integer(value, label, minimum=0):
    require(type(value) is int and minimum <= value <= 2**53 - 1, f"invalid {label}")


def _object(value, label, *, nonempty=True):
    require(isinstance(value, dict) and (bool(value) or not nonempty), f"{label} must be an object")
    canonical_bytes(value)


def _payload(value, label):
    require(isinstance(value, str) and bool(value), f"{label} requires encoded payload")
    try:
        decoded = base64.b64decode(value, validate=True)
        require(bool(decoded) and base64.b64encode(decoded).decode("ascii") == value, "canonical base64 required")
    except (ValueError, UnicodeError) as exc:
        raise LearnerStoreError(f"invalid {label} base64") from exc


RUN_FIELDS = {"schema_version", "role", "learner_id", "backend_session_id", "initialization", "contract",
              "seed", "scope", "catalog", "sampler", "implementation"}
LOGICAL_FIELDS = {"schema_version", "session_id", "role", "theorem_id", "lineage", "completed", "policy_version",
    "adapter_metadata", "value_metadata", "optimizer_metadata", "reference_metadata", "buffer_metadata",
    "event_receipts", "created_at"}
DATA_FIELDS = {"schema_version", "run_sha256", "step", "parent_receipt_sha256", "event", "event_sha256",
    "runtime_receipt", "runtime_receipt_sha256", "batch_refs", "sampler_before_sha256", "sampler_after_sha256",
    "catalog", "catalog_sha256"}
CP_FILES = {"manifest.json", "logical_state.json", "backend_state.json", "data_receipt.json", "sampler_state.json"}
CP_FIELDS = {"schema_version", "run_sha256", "learner_id", "backend_session_id", "step", "parent_checkpoint_sha256",
    "contract_sha256", "catalog_sha256", "data_receipt_sha256", "sampler_sha256", "files"}
RELEASE_FIELDS = {"schema_version", "source", "run_sha256", "contract", "weights_sha256", "data_receipt_sha256",
                  "acceptance", "transfer", "reset"}
CONTRACT_FIELDS = {"backend", "base_sha256", "objective", "hidden_size", "lora_rank", "lora_alpha", "lora_dropout",
                   "target_modules", "value_head", "verified_config"}
PORTABLE_LAYOUT = ".PORTABLE.json"
PORTABLE_COMMIT = ".COMMITTED.json"
PORTABLE_SCHEMA = "reap.learner.package-hardlink-commit.v1"


def _link_noreplace(source, source_name: str, target, target_name: str) -> None:
    """Atomic regular-file link; never emulate with exists()+rename()."""
    # Both directories are open with no-follow handles. The caller just wrote
    # and read back the source regular file, and never mutates it again.
    if os.name == "nt":
        os.link(source.path/source_name, target.path/target_name, follow_symlinks=False)
    else:
        os.link(source_name, target_name, src_dir_fd=source.handle,
                dst_dir_fd=target.handle, follow_symlinks=False)


class LearnerReleaseStore:
    # Subclasses opt into a separate profile; these defaults preserve v1 bytes.
    RUN_SCHEMA = "reap.learner.run.v1"
    DATA_SCHEMA = "reap.learner.data-receipt.v1"
    DATA_FIELDS = DATA_FIELDS
    OBJECTIVE_KIND = OBJECTIVE_KIND
    CONFIG_KEY = "verified_config"
    SNAPSHOT_SCHEMA = "reap.gpu.verified-replay-backend.v1"
    ACCEPTANCE_KIND = "verified-replay-training"

    def __init__(self, root: Path):
        # Keep the un-resolved path: safe_directory rejects each linked ancestor.
        self.root = Path(root)

    def _read(self, kind: str, pin: str, names: set[str]) -> dict[str, bytes]:
        require_sha256(pin, kind + " pin")
        with safe_directory(self.root / kind) as root, root.child(pin) as directory:
            return self._read_package(directory, kind, pin, names)

    @staticmethod
    def _read_package(directory, kind: str, pin: str, names: set[str]) -> dict[str, bytes]:
        present = {p.name for p in directory.path.iterdir()}
        portable = present != names
        if portable:
            require(present == names | {PORTABLE_LAYOUT, PORTABLE_COMMIT},
                    "registry package incomplete or unexpected; never repair it")
            layout = {"schema_version": PORTABLE_SCHEMA, "kind": kind, "pin": pin}
            require(_decode(directory.read(PORTABLE_LAYOUT)) == layout, "portable package layout differs")
            commit = _decode(directory.read(PORTABLE_COMMIT))
            _keys(commit, {"schema_version", "kind", "pin", "files"}, "portable commit")
            require({k: commit[k] for k in layout} == layout, "portable commit identity differs")
        files = {name: directory.read(name) for name in sorted(names)}
        if portable:
            expected = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                        for name, raw in files.items()}
            require(canonical_bytes(commit["files"]) == canonical_bytes(expected), "portable commit checksum mismatch")
        return files

    def _commit_portable(self, root, staging_name: str, kind: str, pin: str, files: dict[str, bytes]) -> None:
        """NFS fallback: exclusive target, immutable links, final commit marker.

        The permanent layout marker is written FIRST: a complete payload set
        without the final marker cannot be mistaken for a legacy package.
        Any interrupted target remains unreadable and is never auto-repaired.
        The staging directory is retained; links share its immutable payloads.
        """
        layout = {"schema_version": PORTABLE_SCHEMA, "kind": kind, "pin": pin}
        commit = {**layout, "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                                      for name, raw in files.items()}}
        with root.child(staging_name) as staging:
            staging.write_new(PORTABLE_COMMIT, canonical_bytes(commit))
            staging.sync()
            require(staging.read(PORTABLE_COMMIT) == canonical_bytes(commit), "portable commit staging differs")
            # mkdir is atomic/exclusive. Even an existing EMPTY directory wins
            # this race; we never populate, replace, or remove another target.
            with root.child(pin, create=True) as target:
                target.write_new(PORTABLE_LAYOUT, canonical_bytes(layout))
                target.sync()
                for name in sorted(files):
                    _link_noreplace(staging, name, target, name)
                target.sync()
                require(all(target.read(name) == raw for name, raw in files.items()), "portable payload checksum mismatch")
                _link_noreplace(staging, PORTABLE_COMMIT, target, PORTABLE_COMMIT)
                target.sync()
                self._read_package(target, kind, pin, set(files))
            root.sync()

    def _commit(self, kind: str, pin: str, files: dict[str, bytes]) -> None:
        """Existing content is verified only; incomplete targets are never repaired."""
        with safe_directory(self.root / kind, create=True) as root:
            opened = False
            try:
                with root.child(pin) as directory:
                    opened = True
                    require(self._read_package(directory, kind, pin, set(files)) == files, "existing content differs")
                    return
            except FileNotFoundError:
                require(not opened, "existing package is incomplete; never repair it")
            staging_name = ".staging-" + uuid.uuid4().hex
            with root.child(staging_name, create=True) as staging:
                for name, raw in sorted(files.items()):
                    staging.write_new(name, raw)
                staging.sync()
                require(all(staging.read(name) == raw for name, raw in files.items()), "staging checksum mismatch")
            try:
                _publish_noreplace(root, staging_name, pin)
            except OSError as exc:
                # Only explicit unsupported-operation errors are known not to
                # have published. I/O errors/timeouts remain unknown: no retry.
                unsupported = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}
                if exc.errno not in unsupported:
                    raise
                self._commit_portable(root, staging_name, kind, pin, files)
                return
            root.sync()

    def _validate_run(self, run: dict) -> None:
        _keys(run, RUN_FIELDS, "learner run")
        require(run["schema_version"] == self.RUN_SCHEMA and run["role"] == "learner", "learner run schema/role mismatch")
        validate_identifier(run["learner_id"], kind="learner_id")
        validate_identifier(run["backend_session_id"], kind="backend_session_id")
        require(type(run["seed"]) is int and 0 <= run["seed"] < 2**63, "invalid signed-63-bit seed")
        for field in ("scope", "sampler", "implementation"):
            _object(run[field], field)
        self._validate_sampler(run["sampler"].get("initial_state"), 0)
        self._validate_contract(run["contract"])
        self._validate_catalog(run["catalog"])
        init = run["initialization"]
        require(isinstance(init, dict), "explicit initialization required")
        if init.get("kind") == "base":
            _keys(init, {"kind"}, "base initialization")
        else:
            _keys(init, {"kind", "release_sha256"}, "release initialization")
            require(init["kind"] == "learner_release", "unsupported initialization kind")
            require_sha256(init["release_sha256"], "initial release")
        canonical_bytes(run)

    @staticmethod
    def _validate_contract(contract):
        _keys(contract, CONTRACT_FIELDS, "verified contract")
        require(contract.get("backend") == "verified-replay" and contract.get("objective") == OBJECTIVE_KIND,
                "only the explicit verified-replay contract is supported")
        require_sha256(contract.get("base_sha256"), "base contract")
        _integer(contract.get("hidden_size"), "hidden_size", 1)
        _object(contract.get("verified_config"), "verified_config")
        require(contract["verified_config"].get("objective") == OBJECTIVE_KIND
                and contract["verified_config"].get("base_tokenizer_sha256") == contract["base_sha256"]
                and contract["verified_config"].get("hidden_size") == contract["hidden_size"], "contract identity mismatch")
        _integer(contract["lora_rank"], "lora rank", 1)
        require(type(contract["lora_alpha"]) in (int, float) and contract["lora_alpha"] > 0
                and type(contract["lora_dropout"]) in (int, float) and 0 <= contract["lora_dropout"] < 1,
                "invalid LoRA alpha/dropout")
        require(isinstance(contract["target_modules"], list) and bool(contract["target_modules"])
                and all(isinstance(x, str) and x for x in contract["target_modules"])
                and len(set(contract["target_modules"])) == len(contract["target_modules"]), "invalid adapter targets")
        require(contract["value_head"] == "linear-silu-linear-categorical-v1", "wrong categorical head contract")
        config = contract["verified_config"]
        _keys(config.get("support"), {"distance_min", "distance_max", "return", "overflow"}, "categorical support")
        support = config["support"]
        require(type(support["distance_min"]) is int and support["distance_min"] == 1
                and type(support["distance_max"]) is int and 2 <= support["distance_max"] <= 4096
                and support["overflow"] == "reject" and support["return"] == "negative_integer_longest_generated_action_branch",
                "wrong categorical support semantics")
        require(config.get("lora") == {"rank": contract["lora_rank"], "alpha": contract["lora_alpha"],
                "dropout": contract["lora_dropout"], "target_modules": contract["target_modules"]}, "nested adapter contract mismatch")
        require(config.get("head") == "linear-silu-linear-categorical", "nested categorical head mismatch")
        canonical_bytes(contract)

    @staticmethod
    def _validate_catalog(catalog):
        require(isinstance(catalog, list) and bool(catalog), "nonempty ordered catalog required")
        seen = set()
        for item in catalog:
            _keys(item, {"dataset_sha256", "profile", "replay_receipt_sha256", "trace_sha256", "source_session_id", "theorem_sha256", "rows", "source_model_release_sha256"}, "catalog item")
            for field in ("dataset_sha256", "replay_receipt_sha256", "trace_sha256", "theorem_sha256"):
                require_sha256(item[field], field)
            require(item["dataset_sha256"] not in seen and item["profile"] == DATA_PROFILE, "duplicate/wrong-profile catalog item")
            seen.add(item["dataset_sha256"])
            validate_identifier(item["source_session_id"], kind="source_session_id")
            _integer(item["rows"], "catalog rows", 1)
            if item["source_model_release_sha256"] is not None:
                require_sha256(item["source_model_release_sha256"], "source model release")

    def create_run(self, run_record: dict) -> str:
        run_record = deepcopy(run_record)
        self._validate_run(run_record)
        self._check_initialization(run_record)
        raw = canonical_bytes(run_record)
        pin = hashlib.sha256(raw).hexdigest()
        self._commit("runs", pin, {"run.json": raw})
        return pin

    def _check_initialization(self, run_record):
        if run_record["initialization"]["kind"] == "learner_release":
            metadata, _ = self.load_release(run_record["initialization"]["release_sha256"])
            require(content_sha256(metadata["contract"]) == content_sha256(run_record["contract"]), "initial release contract differs")

    def load_run(self, pin: str) -> dict:
        raw = self._read("runs", pin, {"run.json"})["run.json"]
        require(hashlib.sha256(raw).hexdigest() == pin, "run content pin mismatch")
        run = _decode(raw)
        self._validate_run(run)
        self._check_initialization(run)
        return run

    @staticmethod
    def _validate_sampler(state: dict, step: int):
        _object(state, "sampler state")
        _integer(state.get("step"), "sampler step")
        _integer(state.get("cursor"), "sampler cursor")
        require(state["step"] == step, "sampler/learner step mismatch")

    def _validate_backend(self, backend, run):
        _keys(backend, {"schema_version", "encoding", "payload", "session_id", self.CONFIG_KEY}, "backend snapshot")
        require(backend["schema_version"] == self.SNAPSHOT_SCHEMA
                and backend["session_id"] == run["backend_session_id"] and backend["encoding"] == "torch-save-base64"
                and content_sha256(backend[self.CONFIG_KEY]) == content_sha256(run["contract"][self.CONFIG_KEY]), "backend snapshot contract mismatch")
        _payload(backend["payload"], "backend")

    def _validate_training_batch(self, run, data, detail, sampler, parent, step):
        event = data["event"]
        refs = data["batch_refs"]
        require(isinstance(refs, list) and 1 <= len(refs) <= 32 and refs == event["samples"], "ordered batch refs mismatch")
        require(isinstance(detail.get("samples"), list) and len(detail["samples"]) == len(refs)
                and canonical_bytes([{"dataset_sha256": item.get("dataset_sha256"), "row": item.get("row")}
                     for item in detail["samples"] if isinstance(item, dict)]) == canonical_bytes(refs),
                "runtime training samples differ from ordered event references")
        catalog = {item["dataset_sha256"]: item for item in data["catalog"]}
        for ref in refs:
            _keys(ref, {"dataset_sha256", "row"}, "batch ref")
            require_sha256(ref["dataset_sha256"], "batch dataset")
            _integer(ref["row"], "batch row")
            require(ref["dataset_sha256"] in catalog and ref["row"] < catalog[ref["dataset_sha256"]]["rows"], "batch ref outside fixed catalog")
        self._validate_sampler(sampler, step)
        before = run["sampler"]["initial_state"] if parent is None else parent["sampler_state"]
        require(data["sampler_before_sha256"] == content_sha256(before)
                and data["sampler_after_sha256"] == content_sha256(sampler), "sampler receipt binding mismatch")
        require(sampler["cursor"] == before["cursor"] + len(refs), "sampler cursor must advance by exact sampled row count")

    def _validate_checkpoint(self, run, run_sha, logical, backend, data, sampler, parent, *, backend_checked=False):
        _keys(logical, LOGICAL_FIELDS, "logical learner snapshot")
        step = logical["policy_version"]
        _integer(step, "committed learner step", 1)
        require(logical["schema_version"] == "reap.gpu.session.v1" and logical["role"] == "learner"
                and logical["session_id"] == run["backend_session_id"] and logical["theorem_id"] is None
                and logical["completed"] is False, "learner logical role/session/completion mismatch")
        for field in ("lineage", "adapter_metadata", "value_metadata", "optimizer_metadata", "reference_metadata", "buffer_metadata", "event_receipts"):
            _object(logical[field], field, nonempty=False)
        require(type(logical["created_at"]) in (int, float), "logical creation time must be numeric")
        buffer = logical["buffer_metadata"]
        _keys(buffer, {"events", "pending_event_ids", "consumed_event_ids"}, "logical event buffer")
        require(buffer["pending_event_ids"] == [], "pending learner events cannot checkpoint")
        _object(buffer["events"], "buffer events")
        require(isinstance(buffer["consumed_event_ids"], list) and len(buffer["consumed_event_ids"]) == step
                and all(isinstance(x, str) for x in buffer["consumed_event_ids"])
                and len(set(buffer["consumed_event_ids"])) == step
                and set(buffer["events"]) == set(logical["event_receipts"]) == set(buffer["consumed_event_ids"]),
                "buffer/receipt event identity mismatch")
        require(type(logical["optimizer_metadata"].get("steps")) is int and logical["optimizer_metadata"]["steps"] == step,
                "logical optimizer/learner step mismatch")
        if not backend_checked:
            self._validate_backend(backend, run)
        _keys(data, self.DATA_FIELDS, "data receipt")
        require(data["schema_version"] == self.DATA_SCHEMA and data["run_sha256"] == run_sha
                and type(data["step"]) is int and data["step"] == step, "data receipt run/step mismatch")
        self._validate_catalog(data["catalog"])
        old_catalog = run["catalog"] if parent is None else parent["data_receipt"]["catalog"]
        require(len(data["catalog"]) >= len(old_catalog)
                and canonical_bytes(data["catalog"][:len(old_catalog)]) == canonical_bytes(old_catalog),
                "catalog may only append; previous descriptors must remain an exact prefix")
        require(data["catalog_sha256"] == content_sha256(data["catalog"]), "current catalog digest mismatch")
        event, receipt = data["event"], data["runtime_receipt"]
        _keys(event, {"kind", "event_id", "session_id", "policy_version", "samples"}, "verified event")
        require(event["kind"] == self.OBJECTIVE_KIND and event["session_id"] == run["backend_session_id"]
                and type(event["policy_version"]) is int and event["policy_version"] == step - 1, "event identity/version mismatch")
        validate_identifier(event["event_id"], kind="event_id")
        _keys(receipt, {"event_id", "applied", "idempotent", "policy_version", "detail"}, "runtime receipt")
        require(receipt["event_id"] == event["event_id"]
                and receipt["applied"] is True and receipt["idempotent"] is False
                and type(receipt["policy_version"]) is int and receipt["policy_version"] == step,
                "runtime receipt is not the exact committed event")
        detail = receipt["detail"]
        _object(detail, "runtime training detail")
        require(detail.get("objective") == self.OBJECTIVE_KIND
                and content_sha256(detail.get("training_config")) == content_sha256(run["contract"][self.CONFIG_KEY])
                and type(detail.get("optimizer_steps")) is int and detail["optimizer_steps"] == step,
                "runtime training objective/config/steps mismatch")
        require(all(detail.get(field) is True for field in ("finite_loss", "finite_gradients", "finite_parameters", "finite_optimizer_state")),
                "runtime receipt lacks successful finite training checks")
        require(data["event_sha256"] == content_sha256(event)
                and data["runtime_receipt_sha256"] == content_sha256(receipt), "event/runtime receipt digest mismatch")
        saved = logical["event_receipts"].get(event["event_id"])
        require(canonical_bytes(saved) == canonical_bytes({"digest": content_sha256(event), "response": receipt}), "logical event receipt mismatch")
        self._validate_training_batch(run, data, detail, sampler, parent, step)
        if parent is None:
            require(step == 1 and data["parent_receipt_sha256"] is None, "first checkpoint must be step one")
            require(buffer["consumed_event_ids"] == [event["event_id"]], "first consumed event differs")
        else:
            require(parent["manifest"]["run_sha256"] == run_sha and step == parent["manifest"]["step"] + 1,
                    "parent checkpoint run/step mismatch")
            require(data["parent_receipt_sha256"] == content_sha256(parent["data_receipt"]), "parent data receipt chain mismatch")
            require(event["event_id"] not in parent["logical_state"]["event_receipts"], "event reused across committed steps")
            require(all(canonical_bytes(logical["event_receipts"].get(k)) == canonical_bytes(v) for k, v in parent["logical_state"]["event_receipts"].items()),
                    "historical event receipts changed")
            require(buffer["consumed_event_ids"] == parent["logical_state"]["buffer_metadata"]["consumed_event_ids"] + [event["event_id"]],
                    "consumed event ordering changed")
        require(len(logical["event_receipts"]) == step, "logical committed event chain has a gap or extra receipt")
        for event_id, saved in logical["event_receipts"].items():
            require(canonical_bytes(buffer["events"][event_id]) == canonical_bytes({"digest": saved["digest"], "status": "consumed",
                    "policy_version": saved["response"]["policy_version"] - 1}), "buffer event does not match committed receipt")
        canonical_bytes(logical)
        return step

    def create_checkpoint(self, run_sha: str, logical_state: dict, backend_state: dict, data_receipt: dict,
                          sampler_state: dict, parent_checkpoint_sha256: str | None = None) -> str:
        logical_state, backend_state, data_receipt, sampler_state = deepcopy(
            (logical_state, backend_state, data_receipt, sampler_state))
        run = self.load_run(run_sha)
        parent = self.load_checkpoint(parent_checkpoint_sha256) if parent_checkpoint_sha256 is not None else None
        if parent is not None:
            parent.pop("backend_state")  # Parent tensor bytes are not needed by receipt-chain validation.
        step = self._validate_checkpoint(run, run_sha, logical_state, backend_state, data_receipt, sampler_state, parent)
        values = {"logical_state.json": logical_state, "backend_state.json": backend_state,
                  "data_receipt.json": data_receipt, "sampler_state.json": sampler_state}
        files = {name: canonical_bytes(value) for name, value in values.items()}
        manifest = {"schema_version": "reap.learner.checkpoint.v1", "run_sha256": run_sha,
            "learner_id": run["learner_id"], "backend_session_id": run["backend_session_id"], "step": step,
            "parent_checkpoint_sha256": parent_checkpoint_sha256, "contract_sha256": content_sha256(run["contract"]),
            "catalog_sha256": content_sha256(data_receipt["catalog"]), "data_receipt_sha256": content_sha256(data_receipt),
            "sampler_sha256": content_sha256(sampler_state),
            "files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()}}
        pin = content_sha256(manifest)
        self._commit("checkpoints", pin, {"manifest.json": canonical_bytes(manifest), **files})
        return pin

    def load_checkpoint(self, pin: str) -> CheckpointPackage:
        chain = []
        seen = set()
        current = pin
        # Iterative parent validation avoids Python recursion limits for long runs.
        while current is not None:
            require(current not in seen, "checkpoint parent cycle")
            seen.add(current)
            files = self._read("checkpoints", current, CP_FILES)
            manifest = _decode(files.pop("manifest.json"))
            _keys(manifest, CP_FIELDS, "checkpoint manifest")
            require(content_sha256(manifest) == current and manifest["schema_version"] == "reap.learner.checkpoint.v1", "checkpoint manifest pin/schema mismatch")
            _keys(manifest["files"], CP_FILES - {"manifest.json"}, "checkpoint file manifest")
            for name, raw in files.items():
                _keys(manifest["files"][name], {"bytes", "sha256"}, "checkpoint file metadata")
                _integer(manifest["files"][name]["bytes"], "file byte count")
                require_sha256(manifest["files"][name]["sha256"], "file checksum")
                require(manifest["files"][name] == {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, "checkpoint file checksum mismatch")
            package = {name[:-5]: _decode(raw) for name, raw in files.items()}
            package.update(manifest=manifest, run=self.load_run(manifest["run_sha256"]))
            self._validate_backend(package["backend_state"], package["run"])
            if chain:
                # Verify every ancestor fully, but do not retain all its large
                # base64 tensor snapshots in memory simultaneously.
                package["backend_state"] = None
            chain.append(package)
            current = manifest["parent_checkpoint_sha256"]
            if current is not None:
                require_sha256(current, "parent checkpoint")
        parent = None
        for package in reversed(chain):
            m, run = package["manifest"], package["run"]
            step = self._validate_checkpoint(run, m["run_sha256"], package["logical_state"], package["backend_state"],
                                            package["data_receipt"], package["sampler_state"], parent, backend_checked=True)
            require(type(m["step"]) is int and m["step"] == step and m["learner_id"] == run["learner_id"] and m["backend_session_id"] == run["backend_session_id"]
                    and m["contract_sha256"] == content_sha256(run["contract"]) and m["catalog_sha256"] == content_sha256(package["data_receipt"]["catalog"])
                    and m["data_receipt_sha256"] == content_sha256(package["data_receipt"])
                    and m["sampler_sha256"] == content_sha256(package["sampler_state"]), "checkpoint summary/contract mismatch")
            parent = package
        require(bool(chain), "checkpoint pin required")
        return chain[0]

    @staticmethod
    def _validate_weights(weights, contract):
        _keys(weights, {"contract", "encoding", "payload"}, "parameter weights")
        require(content_sha256(weights["contract"]) == content_sha256(contract) and weights["encoding"] == "torch-save-base64", "release weights contract/encoding mismatch")
        _payload(weights["payload"], "parameter weights")

    def publish(self, checkpoint_pin: str, weights: dict) -> str:
        weights = deepcopy(weights)
        cp = self.load_checkpoint(checkpoint_pin)
        self._validate_weights(weights, cp["run"]["contract"])
        m = cp["manifest"]
        metadata = {"schema_version": "reap.learner.release.v1",
            "source": {"source_kind": "learner_checkpoint", "learner_id": m["learner_id"], "learner_step": m["step"], "checkpoint_sha256": checkpoint_pin},
            "run_sha256": m["run_sha256"], "contract": cp["run"]["contract"], "weights_sha256": content_sha256(weights),
            "data_receipt_sha256": m["data_receipt_sha256"], "acceptance": {"kind": self.ACCEPTANCE_KIND, "committed_step": m["step"]},
            "transfer": ["adapter", "value_head"], "reset": ["optimizer", "rng", "buffer", "policy_version", "event_receipts"]}
        pin = content_sha256(metadata)
        self._commit("releases", pin, {"metadata.json": canonical_bytes(metadata), "weights.json": canonical_bytes(weights)})
        return pin

    def load_release(self, pin: str) -> tuple[dict, dict]:
        files = self._read("releases", pin, {"metadata.json", "weights.json"})
        metadata, weights = _decode(files["metadata.json"]), _decode(files["weights.json"])
        _keys(metadata, RELEASE_FIELDS, "release metadata")
        require(content_sha256(metadata) == pin and metadata["schema_version"] == "reap.learner.release.v1", "release pin/schema mismatch")
        source = metadata["source"]
        _keys(source, {"source_kind", "learner_id", "learner_step", "checkpoint_sha256"}, "release source")
        _integer(source["learner_step"], "released learner step", 1)
        cp = self.load_checkpoint(source["checkpoint_sha256"])
        m = cp["manifest"]
        require(source == {"source_kind": "learner_checkpoint", "learner_id": m["learner_id"], "learner_step": m["step"], "checkpoint_sha256": source["checkpoint_sha256"]}
                and metadata["run_sha256"] == m["run_sha256"]
                and content_sha256(metadata["contract"]) == content_sha256(cp["run"]["contract"])
                and metadata["data_receipt_sha256"] == m["data_receipt_sha256"], "release source/contract mismatch")
        require(canonical_bytes(metadata["acceptance"]) == canonical_bytes({"kind": self.ACCEPTANCE_KIND, "committed_step": m["step"]})
                and metadata["transfer"] == ["adapter", "value_head"]
                and metadata["reset"] == ["optimizer", "rng", "buffer", "policy_version", "event_receipts"], "release acceptance/transfer mismatch")
        self._validate_weights(weights, metadata["contract"])
        require(metadata["weights_sha256"] == content_sha256(weights), "release weights checksum mismatch")
        return {**metadata, "model_release_sha256": pin}, weights
