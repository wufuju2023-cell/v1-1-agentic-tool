"""Thread-safe logical Session state independent of the GPU backend."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Iterator

from .errors import DuplicateSessionError, SessionNotFoundError
from .identifiers import validate_identifier


@dataclass
class SessionState:
    session_id: str
    role: str = "theorem"
    theorem_id: str | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    policy_version: int = 0
    adapter_metadata: dict[str, Any] = field(default_factory=dict)
    value_metadata: dict[str, Any] = field(default_factory=dict)
    optimizer_metadata: dict[str, Any] = field(default_factory=dict)
    reference_metadata: dict[str, Any] = field(default_factory=dict)
    buffer_metadata: dict[str, Any] = field(
        default_factory=lambda: {"events": {}, "pending_event_ids": [], "consumed_event_ids": []}
    )
    event_receipts: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def snapshot(self) -> dict[str, Any]:
        result = {
            "schema_version": "reap.gpu.session.v1",
            "session_id": self.session_id,
            "theorem_id": self.theorem_id,
            "lineage": deepcopy(self.lineage),
            "completed": self.completed,
            "policy_version": self.policy_version,
            "adapter_metadata": deepcopy(self.adapter_metadata),
            "value_metadata": deepcopy(self.value_metadata),
            "optimizer_metadata": deepcopy(self.optimizer_metadata),
            "reference_metadata": deepcopy(self.reference_metadata),
            "buffer_metadata": deepcopy(self.buffer_metadata),
            "event_receipts": deepcopy(self.event_receipts),
            "created_at": self.created_at,
        }
        # Preserve old v1 theorem snapshots byte-for-byte in the default mode.
        if self.role != "theorem":
            result["role"] = self.role
        return result

    @classmethod
    def from_snapshot(cls, raw: dict[str, Any]) -> "SessionState":
        if raw.get("schema_version") != "reap.gpu.session.v1":
            raise ValueError("unsupported session snapshot schema")
        session_id = validate_identifier(raw.get("session_id"), kind="session_id")
        role = raw.get("role", "theorem")
        if not isinstance(role, str) or role not in {"theorem", "learner", "actor"}:
            raise ValueError("invalid session role")
        version = raw.get("policy_version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 0:
            raise ValueError("invalid policy_version in snapshot")
        theorem_id = raw.get("theorem_id")
        if theorem_id is not None:
            validate_identifier(theorem_id, kind="theorem_id")
        if role == "learner" and (theorem_id is not None or raw.get("completed", False)):
            raise ValueError("learner cannot carry a theorem or completed-proof identity")
        if type(raw.get("completed", False)) is not bool or not isinstance(raw.get("lineage", {}), dict):
            raise ValueError("invalid snapshot completion/lineage metadata")
        if role == "actor" and (theorem_id is None or not raw.get("lineage", {}).get("model_release_sha256")):
            raise ValueError("actor requires theorem and explicit learner release lineage")
        return cls(
            session_id=session_id,
            role=role,
            theorem_id=raw.get("theorem_id"),
            lineage=deepcopy(raw.get("lineage") or {}),
            completed=bool(raw.get("completed", False)),
            policy_version=version,
            adapter_metadata=deepcopy(raw.get("adapter_metadata") or {}),
            value_metadata=deepcopy(raw.get("value_metadata") or {}),
            optimizer_metadata=deepcopy(raw.get("optimizer_metadata") or {}),
            reference_metadata=deepcopy(raw.get("reference_metadata") or {}),
            buffer_metadata=deepcopy(raw.get("buffer_metadata") or {}),
            event_receipts=deepcopy(raw.get("event_receipts") or {}),
            created_at=float(raw.get("created_at", time.time())),
        )


class SessionStore:
    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._states: dict[str, SessionState] = {}
        self._locks: dict[str, threading.RLock] = {}

    def create(self, session_id: str, metadata: dict[str, dict[str, Any]]) -> SessionState:
        session_id = validate_identifier(session_id, kind="session_id")
        with self._guard:
            if session_id in self._states:
                raise DuplicateSessionError(f"session already exists: {session_id}")
            state = SessionState(
                session_id=session_id,
                role=metadata.get("role", "theorem"),
                theorem_id=metadata.get("theorem_id"),
                lineage=deepcopy(metadata.get("lineage") or {}),
                adapter_metadata=deepcopy(metadata.get("adapter") or {}),
                value_metadata=deepcopy(metadata.get("value") or {}),
                optimizer_metadata=deepcopy(metadata.get("optimizer") or {}),
                reference_metadata=deepcopy(metadata.get("reference") or {}),
            )
            self._states[session_id] = state
            self._locks[session_id] = threading.RLock()
            return deepcopy(state)

    def delete(self, session_id: str) -> None:
        session_id = validate_identifier(session_id, kind="session_id")
        with self._guard:
            if session_id not in self._states:
                raise SessionNotFoundError(f"session not found: {session_id}")
            del self._states[session_id]
            del self._locks[session_id]

    def get(self, session_id: str) -> SessionState:
        session_id = validate_identifier(session_id, kind="session_id")
        with self._guard:
            try:
                return deepcopy(self._states[session_id])
            except KeyError as exc:
                raise SessionNotFoundError(f"session not found: {session_id}") from exc

    @contextmanager
    def locked(self, session_id: str) -> Iterator[SessionState]:
        session_id = validate_identifier(session_id, kind="session_id")
        with self._guard:
            try:
                lock = self._locks[session_id]
            except KeyError as exc:
                raise SessionNotFoundError(f"session not found: {session_id}") from exc
        with lock:
            with self._guard:
                state = self._states[session_id]
            yield state

    def replace_locked(self, session_id: str, state: SessionState) -> None:
        """Replace state while the caller holds this session's lock."""
        if state.session_id != session_id:
            raise ValueError("snapshot session id mismatch")
        with self._guard:
            if session_id not in self._states:
                raise SessionNotFoundError(f"session not found: {session_id}")
            self._states[session_id] = state
