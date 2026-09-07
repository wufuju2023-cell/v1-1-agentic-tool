"""Contracts for verified successful-trajectory replay, separate from search Q.

An event supplies content-addressed dataset row references, never its own reward
or return label. The configured loader must validate the complete Lean replay
bundle. CPU-only tests may inject a fixture loader; production uses the strict
verified_trajectory loader, not a caller-selected implementation.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from .identifiers import validate_identifier

OBJECTIVE_KIND = "verified_success_replay"
DATA_PROFILE = "verified-generated-action-negative-longest-branch-v1"
VALUE_SEMANTICS = "verified_negative_longest_branch_categorical.v1"
MAX_BATCH_SAMPLES = 32
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def validate_support(max_distance: Any) -> int:
    # Explicit representational bound, not a search budget or theorem claim.
    return integer(max_distance, "categorical value max_distance", 2, 4096)


def prepare_verified_event(event: dict[str, Any], *, session_id: str,
                           policy_version: int, max_distance: int,
                           load_dataset: Callable[[str], dict]) -> dict[str, Any]:
    max_distance = validate_support(max_distance)
    required = {"kind", "event_id", "session_id", "policy_version", "samples"}
    if not isinstance(event, dict) or set(event) != required or event["kind"] != OBJECTIVE_KIND:
        raise ValueError("verified replay accepts only dataset row references, not client labels")
    if event["session_id"] != session_id:
        raise ValueError("verified replay event session mismatch")
    validate_identifier(event["event_id"], kind="event_id")
    if integer(event["policy_version"], "policy_version", 0, 2**53-1) != policy_version:
        raise ValueError("verified replay event policy version mismatch")
    refs = event["samples"]
    if not isinstance(refs, list) or not 1 <= len(refs) <= MAX_BATCH_SAMPLES:
        raise ValueError("verified replay requires 1..32 sample references")
    datasets: dict[str, dict] = {}
    prepared = []
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"dataset_sha256", "row"}:
            raise ValueError("sample reference requires only dataset_sha256 and row")
        digest = ref["dataset_sha256"]
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError("verified dataset requires a lowercase SHA256 pin")
        if digest not in datasets:
            datasets[digest] = load_dataset(digest)
        dataset = datasets[digest]
        if not isinstance(dataset, dict) or dataset.get("profile") != DATA_PROFILE:
            raise ValueError("verified dataset profile mismatch")
        rows = dataset.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("verified dataset has no successful action rows")
        index = integer(ref["row"], "dataset row", 0, len(rows)-1)
        row = rows[index]
        value = row.get("return")
        if type(value) is not int or not -max_distance <= value <= -1:
            raise ValueError("verified return is outside categorical support; never clip or relabel")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("prompt", "tactic")):
            raise ValueError("verified action requires its original prompt and tactic")
        behavior_version = integer(row.get("policy_version"), "source behavior version", 0, 2**53-1)
        # The source actor and this learner have distinct version counters.
        # CE replay is off-policy; no PPO ratio or equality with learner version.
        prepared.append({"dataset_sha256": digest, "row": index,
            "prompt": row["prompt"], "tactic": row["tactic"], "return": value,
            "distance": -value, "value_class": -value-1,
            "source_session_id": dataset.get("session_id"),
            "source_tree_id": dataset.get("tree_id"),
            "source_theorem_sha256": dataset.get("theorem_sha256"),
            "source_policy_version": behavior_version, "node_index": row.get("node_index")})
    return {"event_id": event["event_id"], "session_id": session_id,
            "policy_version": policy_version, "samples": prepared,
            "sample_weight": 1.0/len(prepared), "value_semantics": VALUE_SEMANTICS,
            "duplicates": "explicit sampling with replacement; each occurrence has equal weight"}
