"""Pure contracts/sampling for explicit 9 replay + 1 human Mathlib rows.

Load callbacks are fixed by the operator/backend, never by an event. Production
must use the two different full-evidence loaders. This module does not train,
tokenize, publish, append a catalog, or commit a sampler step. The caller commits
the returned next state only with its successful learner checkpoint; calling a
pure sampler twice with the same input intentionally returns the same batch.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Callable

from .identifiers import validate_identifier
from .verified_objective import (
    DATA_PROFILE as REPLAY_PROFILE, SHA256, VALUE_SEMANTICS, integer,
    prepare_verified_event, validate_support,
)

OBJECTIVE_KIND = "verified_replay_mathlib_sft"
SFT_PROFILE = "mathlib_sft_linear_negative_remaining_actions_v1"
SOURCE_COUNTS = {"replay": 9, "mathlib_sft": 1}
BATCH_SIZE = 10
SAMPLE_WEIGHT = 1.0 / BATCH_SIZE
TOKENIZATION = "canonical_tactic_separate_no_special_tokens_append_exactly_one_EOS"
SAMPLER_SCHEMA = "reap.mixed-sampler.v1"
STATE_SCHEMA = "reap.mixed-sampler-state.v1"
MAX_INTEGER = 2**53 - 1
LoadDataset = Callable[[str], dict]


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _pin(value: Any) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("dataset requires a lowercase SHA256 pin")
    return value


def _source(value: Any) -> str:
    if not isinstance(value, str) or value not in SOURCE_COUNTS:
        raise ValueError("source must be replay or mathlib_sft")
    return value


def _rows(dataset: Any, source: str) -> list:
    expected = REPLAY_PROFILE if source == "replay" else SFT_PROFILE
    if not isinstance(dataset, dict) or dataset.get("profile") != expected:
        raise ValueError("mixed dataset profile does not match its declared source")
    if source == "mathlib_sft" and (
            dataset.get("source_kind") != "mathlib_sft"
            or "source_policy_version" not in dataset
            or dataset["source_policy_version"] is not None):
        raise ValueError("Mathlib SFT cannot have invented actor behavior provenance")
    rows = dataset.get("rows")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("mixed dataset requires nonempty action rows")
    return rows


def _sft_row(dataset: dict, digest: str, index: int, max_distance: int) -> dict:
    row = _rows(dataset, "mathlib_sft")[index]
    if set(row) != {"row", "state", "tactic", "next_state", "prompt", "return"}:
        raise ValueError("Mathlib row fields cannot contain generated actor fields")
    if integer(row["row"], "Mathlib row index", 0, MAX_INTEGER) != index:
        raise ValueError("Mathlib row index differs from the pinned row")
    value = row["return"]
    if type(value) is not int or not -max_distance <= value <= -1:
        raise ValueError("verified return is outside categorical support; never clip or relabel")
    if any(not isinstance(row[k], str) or not row[k].strip() for k in ("prompt", "tactic")):
        raise ValueError("Mathlib action requires its original prompt and tactic")
    provenance = dataset.get("source")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Mathlib SFT requires its original source provenance")
    return {"source": "mathlib_sft", "dataset_sha256": digest, "row": index,
        "prompt": row["prompt"], "tactic": row["tactic"], "return": value,
        "distance": -value, "value_class": -value-1,
        "source_policy_version": None, "mathlib_source": deepcopy(provenance)}


def _replay_rows(dataset: dict, digest: str, indices: list[int], max_distance: int) -> list[dict]:
    # Reuse the verified return and behavior-version rules without changing the
    # SFT source into an artificial actor. At most 32 rows per validator call.
    result = []
    for start in range(0, len(indices), 32):
        prepared = prepare_verified_event({"kind": "verified_success_replay",
            "event_id": "mixed-data-validation", "session_id": "mixed-data-validation",
            "policy_version": 0,
            "samples": [{"dataset_sha256": digest, "row": index} for index in indices[start:start+32]]},
            session_id="mixed-data-validation", policy_version=0, max_distance=max_distance,
            load_dataset=lambda _: dataset)
        result.extend({"source": "replay", **row} for row in prepared["samples"])
    return result


def prepare_mixed_event(event: dict[str, Any], *, session_id: str,
                        policy_version: int, max_distance: int,
                        load_replay: LoadDataset, load_mathlib_sft: LoadDataset) -> dict:
    """Resolve one exact 9/1 batch. No untrusted label/text/weight is accepted."""
    max_distance = validate_support(max_distance)
    integer(policy_version, "expected policy_version", 0, MAX_INTEGER)
    validate_identifier(session_id, kind="session_id")
    fields = {"kind", "event_id", "session_id", "policy_version", "samples"}
    if not isinstance(event, dict) or set(event) != fields or event["kind"] != OBJECTIVE_KIND:
        raise ValueError("mixed replay accepts only source/pin/row references, not client labels")
    if event["session_id"] != session_id:
        raise ValueError("mixed replay event session mismatch")
    validate_identifier(event["event_id"], kind="event_id")
    if integer(event["policy_version"], "policy_version", 0, MAX_INTEGER) != policy_version:
        raise ValueError("mixed replay event policy version mismatch")
    refs = event["samples"]
    if not isinstance(refs, list) or len(refs) != BATCH_SIZE:
        raise ValueError("mixed replay requires exactly 10 references")
    counts = {source: 0 for source in SOURCE_COUNTS}
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"source", "dataset_sha256", "row"}:
            raise ValueError("mixed sample requires only source, dataset_sha256 and row")
        counts[_source(ref["source"])] += 1
        _pin(ref["dataset_sha256"])
        integer(ref["row"], "dataset row", 0, MAX_INTEGER)
    if counts != SOURCE_COUNTS:
        raise ValueError("mixed replay requires exactly 9 replay and 1 mathlib_sft rows")
    loaders = {"replay": load_replay, "mathlib_sft": load_mathlib_sft}
    datasets, prepared = {}, []
    for ref in refs:
        source, digest, index = ref["source"], ref["dataset_sha256"], ref["row"]
        key = (source, digest)
        if key not in datasets:
            datasets[key] = loaders[source](digest)
        dataset = datasets[key]
        rows = _rows(dataset, source)
        integer(index, "dataset row", 0, len(rows)-1)
        prepared.append(_replay_rows(dataset, digest, [index], max_distance)[0]
            if source == "replay" else _sft_row(dataset, digest, index, max_distance))
    return {"event_id": event["event_id"], "session_id": session_id,
        "policy_version": policy_version, "samples": prepared,
        "source_counts": counts, "sample_weight": SAMPLE_WEIGHT,
        "value_semantics": VALUE_SEMANTICS, "tokenization": TOKENIZATION,
        "duplicates": "explicit cyclic sampling with replacement; every occurrence has weight 1/10"}


def _validated_config(config: Any) -> dict:
    fields = {"schema_version", "kind", "objective", "seed", "max_distance", "catalog"}
    if not isinstance(config, dict) or set(config) != fields or (
            config["schema_version"] != SAMPLER_SCHEMA or config["kind"] != "seeded_cyclic_9_1_v1"
            or config["objective"] != OBJECTIVE_KIND):
        raise ValueError("mixed sampler configuration schema mismatch")
    integer(config["seed"], "sampler seed", 0, MAX_INTEGER)
    validate_support(config["max_distance"])
    catalog = config["catalog"]
    if not isinstance(catalog, dict) or set(catalog) != set(SOURCE_COUNTS):
        raise ValueError("mixed sampler requires both fixed source catalogs")
    all_pins = set()
    for source in SOURCE_COUNTS:
        entries = catalog[source]
        if not isinstance(entries, list) or not entries:
            raise ValueError("mixed sampler cannot fall back when a source catalog is empty")
        total = 0
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"dataset_sha256", "profile", "rows"}:
                raise ValueError("mixed catalog descriptor fields mismatch")
            digest = _pin(entry["dataset_sha256"])
            if digest in all_pins:
                raise ValueError("duplicate/cross-source dataset pin in fixed catalog")
            all_pins.add(digest)
            expected = REPLAY_PROFILE if source == "replay" else SFT_PROFILE
            if entry["profile"] != expected:
                raise ValueError("mixed catalog source profile mismatch")
            total += integer(entry["rows"], "catalog row count", 1, MAX_INTEGER)
        integer(total, "catalog total rows", 1, MAX_INTEGER)
    return config


def make_mixed_sampler(*, replay_pins: list[str], mathlib_sft_pins: list[str],
                       seed: int, max_distance: int, load_replay: LoadDataset,
                       load_mathlib_sft: LoadDataset) -> tuple[dict, dict]:
    """Admit all rows of two ordered seed catalogs; return config + zero state."""
    max_distance = validate_support(max_distance)
    integer(seed, "sampler seed", 0, MAX_INTEGER)
    pins = {"replay": replay_pins, "mathlib_sft": mathlib_sft_pins}
    loaders = {"replay": load_replay, "mathlib_sft": load_mathlib_sft}
    seen = set()
    # Check both catalogs before any loading: no silent empty-SFT fallback.
    for values in pins.values():
        if not isinstance(values, list) or not values:
            raise ValueError("mixed sampler requires nonempty ordered pin lists for both sources")
        for digest in values:
            _pin(digest)
            if digest in seen:
                raise ValueError("duplicate/cross-source dataset pin in fixed catalog")
            seen.add(digest)
    catalog = {source: [] for source in SOURCE_COUNTS}
    for source, values in pins.items():
        for digest in values:
            dataset = loaders[source](digest)
            rows = _rows(dataset, source)
            if source == "replay":
                _replay_rows(dataset, digest, list(range(len(rows))), max_distance)
            else:
                for index in range(len(rows)):
                    _sft_row(dataset, digest, index, max_distance)
            catalog[source].append({"dataset_sha256": digest,
                "profile": dataset["profile"], "rows": len(rows)})
    config = _validated_config({"schema_version": SAMPLER_SCHEMA, "kind": "seeded_cyclic_9_1_v1",
        "objective": OBJECTIVE_KIND, "seed": seed, "max_distance": max_distance, "catalog": catalog})
    return config, {"schema_version": STATE_SCHEMA, "config_sha256": _sha(config),
        "step": 0, "replay_cursor": 0, "mathlib_sft_cursor": 0}


def next_mixed_batch(config: dict, state: dict) -> tuple[list[dict], dict]:
    """Pure sampling with independent cumulative cursors; no global RNG calls.

    A restore must supply its original externally pinned config and committed
    state. The hash detects mismatches; it is not a signature or a durable head.
    No catalog addition/reordering or state reset is an implicit continuation.
    """
    _validated_config(config)
    fields = {"schema_version", "config_sha256", "step", "replay_cursor", "mathlib_sft_cursor"}
    if not isinstance(state, dict) or set(state) != fields or state["schema_version"] != STATE_SCHEMA:
        raise ValueError("mixed sampler state schema mismatch")
    if state["config_sha256"] != _sha(config):
        raise ValueError("mixed sampler state is bound to a different fixed catalog/configuration")
    # Leave space for the next complete batch, keeping JSON integers exact.
    step = integer(state["step"], "sampler step", 0, MAX_INTEGER//9-1)
    for source, count in SOURCE_COUNTS.items():
        cursor = integer(state[source+"_cursor"], source+" cursor", 0, MAX_INTEGER)
        if cursor != step*count:
            raise ValueError("mixed sampler cursor does not match committed step")
    refs = []
    for source, count in SOURCE_COUNTS.items():
        entries = config["catalog"][source]
        total = sum(entry["rows"] for entry in entries)
        offset = int(_sha({"seed": config["seed"], "source": source,
                          "kind": config["kind"]}), 16) % total
        for i in range(count):
            index = (offset+state[source+"_cursor"]+i) % total
            for entry in entries:
                if index < entry["rows"]:
                    refs.append({"source": source, "dataset_sha256": entry["dataset_sha256"], "row": index})
                    break
                index -= entry["rows"]
    after = {**state, "step": step+1,
        "replay_cursor": state["replay_cursor"]+9,
        "mathlib_sft_cursor": state["mathlib_sft_cursor"]+1}
    return refs, after
