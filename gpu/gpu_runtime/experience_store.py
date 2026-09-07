"""Explicit immutable parameter releases, separate from same-session restore.

Acceptance records are trusted operator attestations, not a Lean verifier.
Hashes detect corruption; they do not authenticate an untrusted store owner.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

from .identifiers import validate_identifier
from .snapshot_store import SnapshotStore, _json_bytes, _sha256


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a SHA256 hex digest")
    return value


class ExperienceStore:
    def __init__(self, root: Path):
        self.snapshots = SnapshotStore(root)

    def publish(self, experience_id: str, *, source: dict, acceptance: dict, weights: dict) -> dict:
        experience_id = validate_identifier(experience_id, kind="experience_id")
        if not isinstance(acceptance, dict):
            raise ValueError("explicit acceptance record required")
        if acceptance.get("source") != source:
            raise ValueError("acceptance must bind the exact source snapshot and version")
        if acceptance.get("completed") is not True or acceptance.get("passed") is not True:
            raise ValueError("only completed, accepted artifacts may be published")
        if acceptance.get("kind") not in ("independent-lean", "local-mechanism"):
            raise ValueError("unsupported acceptance kind")
        require_sha256(acceptance.get("evidence_sha256"), "acceptance evidence")
        if weights["contract"]["backend"] != "toy" and acceptance["kind"] != "independent-lean":
            raise ValueError("real parameter releases require independent Lean acceptance")
        metadata = {"schema_version": "reap.gpu.experience.v1", "experience_id": experience_id,
                    "source": deepcopy(source), "acceptance": deepcopy(acceptance),
                    "transfer": ["adapter", "value_head"], "weights_sha256": _sha256(_json_bytes(weights))}
        self.snapshots.create(experience_id, "release", session_state=metadata, backend_state=weights)
        return deepcopy(metadata)

    def load(self, experience_id: str) -> tuple[dict, dict]:
        metadata, weights = self.snapshots.load(experience_id, "release")
        if (metadata.get("schema_version") != "reap.gpu.experience.v1"
                or metadata.get("experience_id") != experience_id
                or metadata.get("weights_sha256") != _sha256(_json_bytes(weights))):
            raise ValueError("experience identity or weights digest mismatch")
        return metadata, weights
