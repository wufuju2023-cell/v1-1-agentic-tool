"""Pure, reviewable contracts for the opt-in search-visit/backup objective.

Reap stores negative distances. For a NON-terminal state whose proof takes L
tactic steps, r is received on the last action, so V=gamma**(L-1), not gamma**L.
Solved states (distance zero), unvisited defaults, and focus-only events do not
enter this training contract. These functions require no torch or GPU.
"""
from __future__ import annotations

import math
from typing import Any

from .identifiers import validate_identifier

VALUE_SEMANTICS = "reap.search_backup_discounted_return.v1"
OBJECTIVE_KIND = "search_visit_backup"
DEFAULT_VALUE_FLOOR = 1e-6
MAX_CANDIDATES = 64


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{label} must be a nonnegative exact JSON integer")
    return value


def validate_gamma(gamma: Any) -> float:
    gamma = finite_number(gamma, "gamma")
    if not 0 < gamma < 1:
        raise ValueError("gamma must lie strictly between zero and one")
    return gamma


def validate_value_floor(value_floor: Any) -> float:
    value_floor = finite_number(value_floor, "value_floor")
    if not 0 < value_floor < 1:
        raise ValueError("value_floor must lie strictly between zero and one")
    return value_floor


def value_to_distance(value: float, gamma: float, *, value_floor: float = DEFAULT_VALUE_FLOOR) -> float:
    """Encode a nonterminal value; zero is a documented finite floor, not solved."""
    gamma, value_floor = validate_gamma(gamma), validate_value_floor(value_floor)
    value = finite_number(value, "value")
    if not 0 <= value <= 1:
        raise ValueError("nonterminal value must be in [0, 1]")
    return 1.0 + math.log(max(value, value_floor)) / math.log(gamma)


def distance_to_value(distance: float, gamma: float, *, value_floor: float = DEFAULT_VALUE_FLOOR) -> float:
    """Decode nonterminal distance; distance zero must be handled by the verifier."""
    gamma, value_floor = validate_gamma(gamma), validate_value_floor(value_floor)
    distance = finite_number(distance, "distance")
    if distance < 1:
        raise ValueError("nonterminal distance must be >= 1; solved/sentinel is not a training target")
    return max(value_floor, math.exp((distance - 1.0) * math.log(gamma)))


def visit_distribution(visits: list[int]) -> list[float]:
    if not visits:
        raise ValueError("visit distribution requires candidates")
    counts = [nonnegative_integer(count, "candidate.visits") for count in visits]
    total = sum(counts)
    if total == 0 or total > 2**53 - 1:
        raise ValueError("visit total must be positive and exactly representable; never substitute prior")
    return [count / total for count in counts]


def weighted_joint_nll(token_logprobs: list[list[float]], visits: list[int]) -> float:
    """Reference formula: -sum_a visit_weight[a] * sum_token log p(token)."""
    if len(token_logprobs) != len(visits):
        raise ValueError("candidate logprobs and visits differ in length")
    weights = visit_distribution(visits)
    terms = []
    for row, weight in zip(token_logprobs, weights):
        if not row:
            raise ValueError("candidate token sequence must be nonempty")
        scores = [finite_number(score, "token logprob") for score in row]
        if any(score > 0 for score in scores):
            raise ValueError("token logprob must be <= 0")
        terms.append(-weight * math.fsum(scores))
    return math.fsum(terms)


def prepare_search_event(event: dict[str, Any], *, session_id: str, policy_version: int,
                         gamma: float, value_floor: float = DEFAULT_VALUE_FLOOR,
                         max_candidates: int = MAX_CANDIDATES) -> dict[str, Any]:
    """Fail closed before any optimizer mutation; caller skips non-trainable nodes.

    Old candidate behavior versions are valid search-distillation provenance,
    not PPO denominators. The event's current version must match the session.
    The backup source is a verified node value_sum/count, never raw edge.value.
    """
    gamma, value_floor = validate_gamma(gamma), validate_value_floor(value_floor)
    if not isinstance(event, dict) or event.get("kind") != OBJECTIVE_KIND:
        raise ValueError("strict backend requires kind=search_visit_backup")
    if event.get("session_id") != session_id:
        raise ValueError("event session_id differs from addressed session")
    validate_identifier(event.get("event_id"), kind="event_id")
    validate_identifier(event.get("tree_id"), kind="tree_id")
    for key in ("step", "node_index", "policy_version"):
        nonnegative_integer(event.get(key), key)
    if event["policy_version"] != policy_version:
        raise ValueError("event policy_version differs from current backend version")
    if validate_gamma(event.get("gamma")) != gamma:
        raise ValueError("event gamma differs from configured Lean/search gamma")
    prompt = event.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("search event requires the original inference prompt")
    reward = finite_number(event.get("reward"), "reward")
    terminal = event.get("terminal_verified")
    if type(terminal) is not bool or reward not in (0, 1) or (reward == 1) != terminal:
        raise ValueError("reward must be 0/1 and equal the verified terminal indicator")
    if event.get("is_focus", False) is not False:
        raise ValueError("focus-only event is not a tactic policy training example")
    backup = event.get("backup")
    if not isinstance(backup, dict) or backup.get("valid") is not True or backup.get("kind") not in ("OR", "AND"):
        raise ValueError("backup must identify a valid OR/AND node statistic")
    count = nonnegative_integer(backup.get("visits"), "backup.visits")
    value_sum = finite_number(backup.get("value_sum"), "backup.value_sum")
    if count == 0:
        raise ValueError("unvisited backup has no training target")
    distance = -value_sum / count
    # Only absorb double-precision rounding at the one-step boundary.
    if distance < 1 - 1e-9:
        raise ValueError("backup is a solved/focus/sentinel statistic, not nonterminal distance")
    rounded_distance = max(1.0, distance)
    raw_target = math.exp((rounded_distance - 1.0) * math.log(gamma))
    value_target = max(value_floor, raw_target)
    candidates = event.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= max_candidates:
        raise ValueError(f"search event requires 1..{max_candidates} candidates")
    prepared = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        tactic = candidate.get("tactic")
        if not isinstance(tactic, str) or not tactic.strip() or tactic in seen:
            raise ValueError("candidate tactics must be nonempty and unique; aggregate duplicate visits first")
        if candidate.get("is_focus", False) is not False:
            raise ValueError("focus child must not be encoded as a tactic candidate")
        seen.add(tactic)
        visits = nonnegative_integer(candidate.get("visits"), "candidate.visits")
        old_version = nonnegative_integer(candidate.get("behavior_version"), "candidate.behavior_version")
        if old_version > policy_version:
            raise ValueError("candidate behavior_version is from the future")
        raw_logprob = finite_number(candidate.get("raw_logprob"), "candidate.raw_logprob")
        if raw_logprob > 0:
            raise ValueError("raw_logprob must be <= 0; prior weight is not logprob")
        prepared.append({"tactic": tactic, "visits": visits, "raw_logprob": raw_logprob,
                         "behavior_version": old_version})
    weights = visit_distribution([item["visits"] for item in prepared])
    for item, weight in zip(prepared, weights):
        item["target_probability"] = weight
    return {
        "event_id": event["event_id"], "session_id": session_id, "tree_id": event["tree_id"],
        "step": event["step"], "node_index": event["node_index"], "policy_version": policy_version,
        "prompt": prompt, "gamma": gamma, "reward": reward, "terminal_verified": terminal,
        "candidates": prepared, "value_target": value_target,
        "value_trace": {"value_semantics": VALUE_SEMANTICS, "source_kind": backup["kind"],
                        "source_value_sum": value_sum, "source_visits": count,
                        "distance_before_roundoff_clip": distance, "distance": rounded_distance,
                        "roundoff_clipped": rounded_distance != distance,
                        "unclipped_target": raw_target, "target": value_target,
                        "value_floor": value_floor, "floor_clipped": raw_target < value_floor,
                        "aggregation": "exp_of_mean_negative_distance_not_mean_return"},
    }
