"""Offline, one-layer Lean action wrapper recognition; no proof repair.

No backend calls this automatically. Callers must retain this complete result,
especially ``raw``, before opting into its canonical text. This function never
extracts a draft from surrounding prose or chooses one of several drafts.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SCHEMA = "reap.lean-action-format.v1"


def _sha(text: str) -> str:
    # Ordinary Unicode text has the standard UTF-8 digest. Surrogatepass keeps
    # even a malformed JSON string auditable without aborting a candidate batch.
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError("nonstandard JSON constant")


def normalize_lean_action(raw: str) -> dict[str, Any]:
    """Recognize one complete wrapper, or return the original string unchanged.

    Accepted wrappers are a sole ``lean``/``lean4`` fenced block, or a complete
    JSON object with exactly one key, ``lean``, whose value is a nonempty string.
    Surrounding ASCII whitespace is permitted. Internal whitespace, ``by``,
    theorem declarations and all proof text are preserved. Extracted content
    is never recursively decoded or repaired. Invalid candidate strings are
    returned, not raised; non-string arguments are programmer errors.
    """
    if not isinstance(raw, str):
        raise TypeError("Lean action formatting requires the original string")
    canonical, format_name, reason = raw, "raw", "no_complete_supported_wrapper"
    candidate = raw.strip(" \t\r\n")
    if not candidate:
        reason = "empty"
    elif candidate.startswith("```"):
        lines = candidate.splitlines(keepends=True)
        opening = re.fullmatch(r"```(lean4|lean)[ \t]*(?:\r\n|\n)", lines[0])
        closing = len(lines) >= 3 and re.fullmatch(r"[ \t]*```[ \t]*", lines[-1])
        if opening and closing:
            body = "".join(lines[1:-1])
            if "```" in body:
                reason = "multiple_or_nested_fences"
            elif not body.strip():
                reason = "empty_fenced_body"
            else:
                canonical, format_name, reason = body, opening.group(1) + "_fence", "single_complete_fence"
        else:
            reason = "unsupported_or_incomplete_fence"
    elif candidate.startswith("{"):
        try:
            parsed = json.loads(candidate, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            reason = "invalid_or_nonunique_json"
        else:
            if type(parsed) is dict and set(parsed) == {"lean"} and isinstance(parsed["lean"], str):
                if parsed["lean"].strip():
                    canonical, format_name, reason = parsed["lean"], "lean_json", "single_complete_lean_object"
                else:
                    reason = "empty_json_lean"
            else:
                reason = "unsupported_json_fields_or_type"
    return {"schema_version": SCHEMA, "raw": raw, "canonical": canonical,
            "format": format_name, "changed": canonical != raw, "reason": reason,
            "raw_sha256": _sha(raw), "canonical_sha256": _sha(canonical),
            "hash_encoding": "utf-8-surrogatepass"}
