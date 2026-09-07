"""Deterministic backend used to test the runtime without torch or a GPU."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any

from .backend import BackendLearnResult
from .errors import DuplicateSessionError, SessionNotFoundError
from .schemas import ChatRequest


@dataclass
class _ToyState:
    learned: bool = False
    optimizer_steps: int = 0
    value_score: float = 0.0


class ToyBackend:
    """Starts with a guaranteed-failing tactic and learns ``trivial``."""

    FAILURE_TACTIC = "fail_if_success trivial"

    def __init__(self, *, compute_delay: float = 0.0) -> None:
        self.compute_delay = float(compute_delay)
        self._states: dict[str, _ToyState] = {}

    def _delay(self) -> None:
        if self.compute_delay > 0:
            time.sleep(self.compute_delay)

    def _state(self, session_id: str) -> _ToyState:
        try:
            return self._states[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(f"backend session not found: {session_id}") from exc

    def create_session(self, session_id: str) -> dict[str, dict[str, Any]]:
        self._delay()
        if session_id in self._states:
            raise DuplicateSessionError(f"backend session already exists: {session_id}")
        self._states[session_id] = _ToyState()
        return {
            "adapter": {"backend": "toy", "mode": "failure"},
            "value": {"head": "toy", "score": 0.0},
            "optimizer": {"kind": "toy", "steps": 0},
            "reference": {"frozen": True, "policy": self.FAILURE_TACTIC},
        }

    def delete_session(self, session_id: str) -> None:
        self._delay()
        if self._states.pop(session_id, None) is None:
            raise SessionNotFoundError(f"backend session not found: {session_id}")

    def policy(self, session_id: str, request: ChatRequest) -> tuple[list[str], list[list[dict[str, Any]]]]:
        self._delay()
        state = self._state(session_id)
        tactic = "trivial" if state.learned else self.FAILURE_TACTIC
        contents = [tactic for _ in range(request.n)]
        logprobs = [
            [{"token": tactic, "logprob": -0.01 if state.learned else -0.5}]
            for _ in range(request.n)
        ]
        return contents, logprobs

    def value(self, session_id: str, request: ChatRequest) -> float:
        del request
        self._delay()
        return self._state(session_id).value_score

    def learn(self, session_id: str, event: dict[str, Any]) -> BackendLearnResult:
        self._delay()
        state = self._state(session_id)
        state.optimizer_steps += 1
        # A negative verifier signal is enough for the deterministic toy to
        # stop proposing its known-bad tactic on the next fresh-root segment.
        if float(event.get("reward", 0.0)) != 0:
            state.learned = True
            state.value_score = 1.0
        return BackendLearnResult(
            adapter_metadata={"backend": "toy", "mode": "trivial" if state.learned else "failure"},
            value_metadata={"head": "toy", "score": state.value_score},
            optimizer_metadata={"kind": "toy", "steps": state.optimizer_steps},
            detail={"learned": state.learned, "optimizer_steps": state.optimizer_steps},
        )

    def export_session(self, session_id: str) -> dict[str, Any]:
        self._delay()
        state = self._state(session_id)
        return {
            "schema_version": "reap.gpu.toy-backend.v1",
            "learned": state.learned,
            "optimizer_steps": state.optimizer_steps,
            "value_score": state.value_score,
        }

    def import_session(self, session_id: str, state: dict[str, Any]) -> None:
        self._delay()
        if state.get("schema_version") != "reap.gpu.toy-backend.v1":
            raise ValueError("unsupported toy backend snapshot schema")
        self._state(session_id)
        self._states[session_id] = _ToyState(
            learned=bool(state.get("learned", False)),
            optimizer_steps=int(state.get("optimizer_steps", 0)),
            value_score=float(state.get("value_score", 0.0)),
        )

    def inspect(self, session_id: str) -> dict[str, Any]:
        """Return a copy for tests and diagnostics; call through the actor."""
        return deepcopy(self.export_session(session_id))

    def experience_contract(self) -> dict[str, Any]:
        return {"backend": "toy", "base": self.FAILURE_TACTIC, "objective": "toy-v1"}

    def experience_weights(self, session_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        if (snapshot.get("schema_version") != "reap.gpu.toy-backend.v1"
                or snapshot.get("experience_contract") != self.experience_contract()):
            raise ValueError("toy experience contract mismatch")
        return {"contract": self.experience_contract(), "adapter": snapshot["learned"],
                "value_head": snapshot["value_score"]}

    def initialize_from_experience(self, session_id: str, weights: dict[str, Any]) -> dict[str, Any]:
        import math
        if (set(weights) != {"contract", "adapter", "value_head"}
                or weights["contract"] != self.experience_contract()
                or type(weights["adapter"]) is not bool
                or type(weights["value_head"]) not in (int, float)
                or not math.isfinite(weights["value_head"])):
            raise ValueError("invalid toy experience")
        state = self._state(session_id)
        if state.optimizer_steps:
            raise ValueError("experience requires a fresh session")
        state.learned = weights["adapter"]
        state.value_score = weights["value_head"]
        return {"adapter": {"backend": "toy", "mode": "trivial" if state.learned else "failure"},
                "value": {"head": "toy", "score": state.value_score}}
