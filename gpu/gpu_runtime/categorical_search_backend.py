"""Opt-in REAL7B search backend with a pinned pretrained 64-bin value head.

This mode is deliberately separate from both the legacy scalar search backend
and ``VerifiedReplayBackend``.  Policy generation, visit-policy training, KL
guarding and snapshot transactionality are inherited unchanged.  Only the
value representation is replaced:

* inference decodes ``softmax(logits)`` at temperature 1 as E[distance] over
  support 1..64 (bin 64 is the saturated >=64 bucket);
* online search backups use two-hot categorical cross entropy for fractional
  distances, clipped to the same support;
* a pretrained (P64) or matched-random-initial (R64) artifact is mandatory and
  is bound by its exact file SHA256, schema, role, shapes and REAL7B feature
  fingerprint before any session can be created.

Loading the offline head does not import its training optimizer.  Every new
online session starts at policy/value version zero with an empty AdamW state.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .frozen_base_manifest import build_manifest
from .search_backend import RealSearchBackend


SNAPSHOT_SCHEMA = "reap.gpu.real-search-categorical-backend.v1"
ARTIFACT_SCHEMA = "new_value_head.categorical-head.v2"
FULL_ARTIFACT_SCHEMA = "new_value_head.categorical-head.v3"
FULL_ARTIFACT_PROFILE = "full-eligible-train-205628-isolated-v1"
BACKEND_KIND = "real-search-categorical"
PLATFORM_LOCAL_BASE_SCHEMA = "reap.real7-platform-local-base.v1"
VALUE_SEMANTICS = "new_value_head.distance_categorical_64_saturated.v1"
DECODE = "softmax-temperature-1-expected-distance-1-to-64-bin64-saturates"
ONLINE_TARGET = "clipped-distance-two-hot-categorical-cross-entropy"
HEAD_KIND = "linear-3584-silu-256-linear-64"
MODEL_REPOSITORY = "FrenzyMath/REAL-Prover"
MODEL_REVISION = "fe76f68d9a88f342cb7b546307c20292fea9cced"
MODEL_DIRECTORY = "REAL-Prover-fe76f68d"
MODEL_WEIGHT_BYTES = 15_231_271_864
SUPPORT_MAX = 64
TRAIN_BOUNDARIES = {20_000, 40_000, 60_000, 80_000}
FULL_TRAIN_ROWS = 205_628
ARTIFACT_ROLES = {"pretrained", "matched-random-initial"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode()


def real7_feature_fingerprint(model_path: str | Path) -> dict[str, Any]:
    """Recompute the exact identity used by feature extraction.

    The lock digest binds the verified official file hashes; the four weight
    names/sizes and selected tokenizer/config hashes reproduce the extractor's
    immutable fingerprint rather than trusting an artifact's model claim.
    """
    root = Path(model_path)
    names = ("config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "generation_config.json",
             "model.safetensors.index.json")
    metadata = [{"path": name, "size": (root/name).stat().st_size,
                 "sha256": _sha256(root/name)} for name in names if (root/name).is_file()]
    weights = [{"path": path.name, "size": path.stat().st_size}
               for path in sorted(root.glob("*.safetensors"))]
    lock_path = root/"reap-model-lock.json"
    if (root.name != MODEL_DIRECTORY or not lock_path.is_file() or not metadata
            or len(weights) != 4 or sum(item["size"] for item in weights) != MODEL_WEIGHT_BYTES):
        raise ValueError("categorical value artifact requires the pinned REAL7B model directory")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("revision") != MODEL_REVISION
            or lock.get("safetensors", {}).get("tensor_count") != 339):
        raise ValueError("categorical value REAL7B verification lock mismatch")
    result = {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION,
              "directory": root.name, "model_lock_sha256": _sha256(lock_path),
              "metadata": metadata, "weights": weights}
    result["sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def distance_two_hot(distance: Any, *, support_max: int = SUPPORT_MAX) -> dict[str, Any]:
    """Project a possibly fractional backup distance onto adjacent bins.

    Bin ``support_max`` is saturated: every finite distance >= support_max is
    assigned to it with weight one.  Values below one are invalid nonterminal
    search targets and are never silently repaired.
    """
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        raise ValueError("categorical distance must be a finite number")
    distance = float(distance)
    if not math.isfinite(distance) or distance < 1:
        raise ValueError("categorical nonterminal distance must be finite and >= 1")
    if type(support_max) is not int or support_max < 2:
        raise ValueError("categorical support maximum must be an integer >= 2")
    clipped = min(distance, float(support_max))
    lower, upper = math.floor(clipped), math.ceil(clipped)
    upper_weight = clipped-lower
    lower_weight = 1.0-upper_weight
    if lower == upper:
        lower_weight, upper_weight = 1.0, 0.0
    weights = [0.0]*support_max
    weights[lower-1] += lower_weight
    if upper_weight:
        weights[upper-1] += upper_weight
    return {"distance": distance, "clipped_distance": clipped,
            "support_min": 1, "support_max": support_max,
            "lower_bin": lower, "upper_bin": upper,
            "lower_weight": lower_weight, "upper_weight": upper_weight,
            "saturated": distance >= support_max, "weights": weights}


class CategoricalSearchBackend(RealSearchBackend):
    """REAL-Prover search/TTT with an immutable offline P64 or R64 head seed."""

    _categorical_value_training = True

    def __init__(self, model_path: str, *, gamma: float,
                 value_artifact: str | Path, value_artifact_sha256: str,
                 value_artifact_role: str, **kwargs: Any) -> None:
        artifact_path = Path(value_artifact).absolute()
        if (not isinstance(value_artifact_sha256, str)
                or SHA256.fullmatch(value_artifact_sha256) is None):
            raise ValueError("categorical value artifact SHA256 must be lowercase hex")
        if value_artifact_role not in ARTIFACT_ROLES:
            raise ValueError("categorical value artifact role must be pretrained or matched-random-initial")
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise ValueError("categorical value artifact must be an existing regular non-link file")
        actual_sha256 = _sha256(artifact_path)
        if actual_sha256 != value_artifact_sha256:
            raise ValueError("categorical value artifact SHA256 mismatch")
        feature_fingerprint = real7_feature_fingerprint(model_path)
        frozen_base_manifest = build_manifest(model_path)
        super().__init__(model_path, gamma=gamma, **kwargs)
        artifact = self.torch.load(artifact_path, map_location="cpu", weights_only=False)
        self._artifact_state, self._artifact_metadata = self._validate_artifact(
            artifact, actual_sha256, value_artifact_role, feature_fingerprint)
        self._frozen_base_manifest = frozen_base_manifest

    def _validate_artifact(self, artifact: Any, digest: str, role: str,
                           feature_fingerprint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        base_fields = {"schema_version", "hidden_size", "classes", "train_rows", "seed",
                  "initial_head_sha256", "feature_model_fingerprint_sha256",
                  "dataset_manifest_sha256", "decode", "online_target", "optimizer",
                  "artifact_role", "state_dict"}
        full = isinstance(artifact, dict) and artifact.get("schema_version") == FULL_ARTIFACT_SCHEMA
        fields = base_fields | ({"profile", "extension_provenance"} if full else set())
        if not isinstance(artifact, dict) or set(artifact) != fields:
            raise ValueError("categorical value artifact fields/schema mismatch")
        valid_scale = ((artifact["schema_version"] == ARTIFACT_SCHEMA
                        and artifact["train_rows"] in TRAIN_BOUNDARIES)
                       or (full and artifact["train_rows"] == FULL_TRAIN_ROWS
                           and artifact.get("profile") == FULL_ARTIFACT_PROFILE))
        if (not valid_scale or artifact["hidden_size"] != 3584
                or artifact["classes"] != SUPPORT_MAX
                or type(artifact["seed"]) is not int or artifact["artifact_role"] != role
                or artifact["decode"] != DECODE or artifact["online_target"] != ONLINE_TARGET):
            raise ValueError("categorical value artifact contract mismatch")
        if full:
            provenance = artifact.get("extension_provenance")
            required = {"extension_feature_manifest_sha256", "dataset_extension_manifest_sha256",
                        "base_feature_manifest_sha256", "base_dataset_manifest_sha256"}
            if (not isinstance(provenance, dict) or set(provenance) != required
                    or any(not isinstance(provenance[key], str)
                           or SHA256.fullmatch(provenance[key]) is None for key in required)
                    or provenance["base_dataset_manifest_sha256"]
                        != artifact["dataset_manifest_sha256"]):
                raise ValueError("full categorical extension provenance mismatch")
        for key in ("initial_head_sha256", "feature_model_fingerprint_sha256",
                    "dataset_manifest_sha256"):
            if not isinstance(artifact[key], str) or SHA256.fullmatch(artifact[key]) is None:
                raise ValueError("categorical value artifact contains an invalid provenance SHA256")
        if artifact["feature_model_fingerprint_sha256"] != feature_fingerprint["sha256"]:
            raise ValueError("categorical value artifact/base feature fingerprint mismatch")
        optimizer = artifact["optimizer"]
        if (not isinstance(optimizer, dict) or set(optimizer) != {"name", "learning_rate", "weight_decay"}
                or optimizer["name"] != "AdamW"
                or isinstance(optimizer["learning_rate"], bool)
                or not isinstance(optimizer["learning_rate"], (int, float))
                or not math.isfinite(float(optimizer["learning_rate"]))
                or float(optimizer["learning_rate"]) <= 0
                or isinstance(optimizer["weight_decay"], bool)
                or not isinstance(optimizer["weight_decay"], (int, float))
                or not math.isfinite(float(optimizer["weight_decay"]))
                or float(optimizer["weight_decay"]) < 0):
            raise ValueError("categorical offline optimizer metadata is invalid")
        state = artifact["state_dict"]
        shapes = {"0.weight": (256, 3584), "0.bias": (256,),
                  "2.weight": (SUPPORT_MAX, 256), "2.bias": (SUPPORT_MAX,)}
        if not isinstance(state, dict) or set(state) != set(shapes):
            raise ValueError("categorical value artifact tensor names mismatch")
        clean = {}
        for name, shape in shapes.items():
            tensor = state[name]
            if (not self.torch.is_tensor(tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != self.torch.float32 or not bool(self.torch.isfinite(tensor).all())):
                raise ValueError("categorical value artifact tensor shape/dtype/finiteness mismatch")
            clean[name] = tensor.detach().cpu().clone()
        state_digest = hashlib.sha256(b"".join(clean[name].numpy().tobytes()
            for name in ("0.weight", "0.bias", "2.weight", "2.bias"))).hexdigest()
        if role == "matched-random-initial" and state_digest != artifact["initial_head_sha256"]:
            raise ValueError("matched random artifact tensors do not match initial_head_sha256")
        metadata = {key: artifact[key] for key in fields-{"state_dict"}}
        metadata.update({"artifact_sha256": digest,
                         "state_dict_sha256": state_digest,
                         "feature_model_fingerprint": feature_fingerprint})
        return clean, metadata

    def _categorical_config(self) -> dict[str, Any]:
        return {"head": HEAD_KIND, "classes": SUPPORT_MAX,
                "decode": DECODE, "online_target": ONLINE_TARGET,
                "temperature": 1.0,
                "support": {"distance_min": 1, "distance_max": SUPPORT_MAX,
                            "bin_64": "all_distances_greater_than_or_equal_to_64"},
                "artifact": self._artifact_metadata,
                "offline_optimizer_loaded": False,
                "online_optimizer_initial_state": "empty_at_policy_version_0"}

    def _search_config(self) -> dict[str, Any]:
        config = super()._search_config()
        config.update({"value_semantics": VALUE_SEMANTICS,
                       "max_distance": SUPPORT_MAX,
                       "head": "categorical-64",
                       "categorical_value": self._categorical_config()})
        return config

    def _value_metadata(self, **extra: Any) -> dict[str, Any]:
        return {"hidden_size": self.hidden_size, **self._search_config(), **extra}

    def create_session(self, session_id: str) -> dict[str, Any]:
        metadata = super().create_session(session_id)
        try:
            session, torch = self.sessions[session_id], self.torch
            # Replace the scalar head and its optimizer before this session can
            # be observed.  Both P64 and R64 execute the same constructor path.
            with self._session_rng(session):
                session.value_head = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size, 256), torch.nn.SiLU(),
                    torch.nn.Linear(256, SUPPORT_MAX)).to(self.device, dtype=torch.float32)
                session.value_head.load_state_dict(self._artifact_state, strict=True)
                session.optimizer = torch.optim.AdamW([
                    {"params": self._adapter_parameters(session_id), "lr": self.learning_rate},
                    {"params": list(session.value_head.parameters()), "lr": self.value_learning_rate}])
            if session.optimizer_steps != 0 or session.examples_seen != 0 or session.optimizer.state:
                raise RuntimeError("categorical offline load must start at v0 with empty online optimizer")
            copied = session.value_head.state_dict()
            if any(not torch.equal(copied[name].detach().cpu(), tensor)
                   for name, tensor in self._artifact_state.items()):
                raise RuntimeError("categorical value artifact copy verification failed")
            metadata["adapter"]["objective"] = "search_visit_backup"
            metadata["value"] = self._value_metadata(offline_artifact_loaded=True)
            metadata["optimizer"] = {"kind": "AdamW", "steps": 0,
                                     "state_empty": True, "offline_optimizer_loaded": False}
            return metadata
        except BaseException:
            self.delete_session(session_id)
            raise

    def value(self, session_id: str, request: Any) -> float:
        session, torch = self._activate(session_id), self.torch
        self.model.eval(); session.value_head.eval()
        with torch.no_grad():
            output = self.model(**self._encoded_prompt(request), output_hidden_states=True,
                                logits_to_keep=1, use_cache=False, return_dict=True)
            logits = session.value_head(output.hidden_states[-1][:, -1, :].float())
            if logits.shape != (1, SUPPORT_MAX):
                raise RuntimeError("categorical value logits shape mismatch")
            self._require_finite(logits, "categorical value logits")
            support = torch.arange(1, SUPPORT_MAX+1, device=logits.device, dtype=logits.dtype)
            distance = (logits.softmax(-1)*support).sum(-1)
            self._require_finite(distance, "categorical expected distance")
            return min(float(SUPPORT_MAX), max(1.0, float(distance.item())))

    def _categorical_loss(self, session: Any, hidden: Any, distance: float):
        torch = self.torch
        projection = distance_two_hot(distance)
        logits = session.value_head(hidden)
        if logits.shape != (1, SUPPORT_MAX):
            raise RuntimeError("categorical training logits shape mismatch")
        target = torch.tensor([projection["weights"]], device=self.device, dtype=torch.float32)
        log_probabilities = self.functional.log_softmax(logits, dim=-1)
        loss = -(target*log_probabilities).sum(-1).mean()
        support = torch.arange(1, SUPPORT_MAX+1, device=logits.device, dtype=logits.dtype)
        expected = (logits.softmax(-1)*support).sum(-1).squeeze()
        audit = {key: value for key, value in projection.items() if key != "weights"}
        audit.update({"prediction": float(expected.detach().cpu()),
                      "target": projection["clipped_distance"],
                      "loss_kind": "two_hot_categorical_cross_entropy",
                      "target_nonzero": [{"bin": index+1, "weight": weight}
                                         for index, weight in enumerate(projection["weights"]) if weight]})
        return expected, target, loss, audit

    def _search_value_loss(self, session: Any, hidden: Any,
                           example: dict[str, Any]):
        return self._categorical_loss(session, hidden, example["value_trace"]["distance"])

    def _success_value_loss(self, session: Any, hidden: Any,
                            row: dict[str, Any]):
        return self._categorical_loss(session, hidden, -row["return"])

    def export_session(self, session_id: str) -> dict[str, Any]:
        state = super().export_session(session_id)
        return {**state, "schema_version": SNAPSHOT_SCHEMA, "session_id": session_id,
                "categorical_search_config": self._search_config()}

    def import_session(self, session_id: str, state: dict[str, Any]) -> None:
        if (state.get("schema_version") != SNAPSHOT_SCHEMA or state.get("session_id") != session_id
                or state.get("categorical_search_config") != self._search_config()):
            raise ValueError("categorical snapshot session/artifact/value semantics mismatch")
        legacy = {**state, "schema_version": "reap.gpu.real-search-backend.v1"}
        super().import_session(session_id, legacy)

    def experience_contract(self) -> dict[str, Any]:
        inherited = super().experience_contract()
        # The inherited digest is deliberately retained only for same-host
        # auditing: it includes model.config.to_dict() and named buffers, so it
        # is not a cross-platform transfer identity.
        self._platform_local_base_sha256 = inherited["base_sha256"]
        return {**inherited, "base_sha256": self._frozen_base_manifest["sha256"],
                "backend": BACKEND_KIND,
                "value_head": "linear-silu-linear-categorical-64-v1",
                "categorical_search_config": self._search_config()}

    def platform_local_base_identity(self) -> dict[str, Any]:
        """Return an audit-only digest; callers must not use it for handoff."""
        self.experience_contract()
        return {"schema_version": PLATFORM_LOCAL_BASE_SCHEMA,
                "portable_base_sha256": self._frozen_base_manifest["sha256"],
                "platform_local_base_sha256": self._platform_local_base_sha256}

    def _check_experience_snapshot(self, session_id: str, snapshot: dict[str, Any]) -> None:
        if (snapshot.get("schema_version") != SNAPSHOT_SCHEMA
                or snapshot.get("session_id") != session_id
                or snapshot.get("categorical_search_config") != self._search_config()
                or snapshot.get("experience_contract") != self.experience_contract()):
            raise ValueError("categorical experience source/session/artifact mismatch")

    def initialize_from_experience(self, session_id: str, weights: dict[str, Any]) -> dict[str, Any]:
        metadata = super().initialize_from_experience(session_id, weights)
        metadata["adapter"]["objective"] = "search_visit_backup"
        metadata["value"] = self._value_metadata(initialized_from_experience=True)
        return metadata
