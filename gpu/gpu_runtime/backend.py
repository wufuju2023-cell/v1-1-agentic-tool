"""Backend protocol shared by the toy and future GPU implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .schemas import ChatRequest


@dataclass(frozen=True)
class BackendLearnResult:
    adapter_metadata: dict[str, Any]
    value_metadata: dict[str, Any]
    optimizer_metadata: dict[str, Any]
    detail: dict[str, Any]


class Backend(Protocol):
    def create_session(self, session_id: str) -> dict[str, dict[str, Any]]:
        """Create backend state and return adapter/value/optimizer/ref metadata."""

    def delete_session(self, session_id: str) -> None:
        """Delete all mutable backend state for a session."""

    def policy(self, session_id: str, request: ChatRequest) -> tuple[list[str], list[list[dict[str, Any]]]]:
        """Return candidate texts and token-level logprob entries."""

    def value(self, session_id: str, request: ChatRequest) -> float:
        """Return the value score that Reap expects inside JSON content."""

    def learn(self, session_id: str, event: dict[str, Any]) -> BackendLearnResult:
        """Apply exactly one idempotency-guarded runtime update."""

    def export_session(self, session_id: str) -> dict[str, Any]:
        """Return JSON-serializable mutable backend state."""

    def import_session(self, session_id: str, state: dict[str, Any]) -> None:
        """Replace mutable backend state from a verified snapshot."""

    def experience_contract(self) -> dict[str, Any]:
        """Return path-independent base/model/objective compatibility identity."""

    def experience_weights(self, session_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Extract only approved parameters from an identity-checked source snapshot."""

    def initialize_from_experience(self, session_id: str, weights: dict[str, Any]) -> dict[str, Any]:
        """Initialize a fresh destination; never import optimizer/RNG/counters."""
