"""Identifier validation and containment checks."""

from __future__ import annotations

from pathlib import Path
import re

from .errors import InvalidIdentifierError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def validate_identifier(value: str, *, kind: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise InvalidIdentifierError(f"invalid {kind}: {value!r}")
    if value in {".", ".."}:
        raise InvalidIdentifierError(f"invalid {kind}: {value!r}")
    return value


def contained_path(root: Path, *parts: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise InvalidIdentifierError(f"path escapes snapshot root: {candidate}") from exc
    return candidate
