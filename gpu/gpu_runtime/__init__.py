"""Session-isolated GPU runtime primitives for REAP training.

The default backend is deliberately a standard-library-only toy.  A real
PyTorch/Transformers backend can implement :class:`Backend` without making
those heavy dependencies import-time requirements for the control plane.
"""

from .actor import GpuActor
from .backend import Backend, BackendLearnResult
from .errors import (
    DuplicateSessionError,
    EventConflictError,
    InvalidIdentifierError,
    SessionNotFoundError,
    SnapshotIntegrityError,
    SnapshotNotFoundError,
    VersionConflictError,
)
from .runtime import GpuRuntime
from .session_store import SessionState, SessionStore
from .snapshot_store import SnapshotStore
from .toy_backend import ToyBackend

# RealProverBackend itself has no import-time torch dependency; the heavy
# imports occur only when it is instantiated on the GPU host.
from .real_backend import RealProverBackend

__all__ = [
    "Backend",
    "BackendLearnResult",
    "DuplicateSessionError",
    "EventConflictError",
    "GpuActor",
    "GpuRuntime",
    "InvalidIdentifierError",
    "RealProverBackend",
    "SessionNotFoundError",
    "SessionState",
    "SessionStore",
    "SnapshotIntegrityError",
    "SnapshotNotFoundError",
    "SnapshotStore",
    "ToyBackend",
    "VersionConflictError",
]
