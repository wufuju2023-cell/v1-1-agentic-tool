"""A single-worker FIFO actor for all mutable GPU operations."""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
from time import monotonic
from typing import Any, Callable


@dataclass
class _WorkItem:
    function: Callable[[], Any]
    submitted_at: float
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class GpuActor:
    """Serialize backend access even when HTTP/session callers are concurrent."""

    def __init__(self, *, name: str = "reap-gpu-actor") -> None:
        self._queue: queue.Queue[_WorkItem | None] = queue.Queue()
        self._metrics_guard = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.completed = 0
        self._submitted = 0
        self._started = 0
        self._failed = 0
        self._queued = 0
        self._max_queued = 0
        self._queue_wait_seconds = 0.0
        self._max_queue_wait_seconds = 0.0
        self._execution_seconds = 0.0
        self._max_execution_seconds = 0.0
        self._active_since: float | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                with self._metrics_guard:
                    started_at = monotonic()
                    queue_wait = started_at - item.submitted_at
                    self._queued -= 1
                    self._started += 1
                    self._queue_wait_seconds += queue_wait
                    self._max_queue_wait_seconds = max(self._max_queue_wait_seconds, queue_wait)
                    self._active_since = started_at
                    self._active += 1
                    self.max_active = max(self.max_active, self._active)
                try:
                    item.result = item.function()
                except BaseException as exc:  # propagate the exact backend error to the caller
                    item.error = exc
                finally:
                    with self._metrics_guard:
                        execution_seconds = monotonic() - started_at
                        self._execution_seconds += execution_seconds
                        self._max_execution_seconds = max(self._max_execution_seconds, execution_seconds)
                        self._failed += int(item.error is not None)
                        self._active_since = None
                        self._active -= 1
                        self.completed += 1
                    item.done.set()
            finally:
                self._queue.task_done()

    def submit(self, function: Callable[[], Any]) -> Any:
        if threading.current_thread() is self._thread:
            raise RuntimeError("GPU actor cannot submit recursively from its worker")
        # Acceptance and the shutdown sentinel use the same lock: no accepted
        # request may land behind the sentinel and wait forever for a receipt.
        with self._metrics_guard:
            if self._closed:
                raise RuntimeError("GPU actor is closed")
            item = _WorkItem(function=function, submitted_at=monotonic())
            self._queue.put(item)
            self._submitted += 1
            self._queued += 1
            self._max_queued = max(self._max_queued, self._queued)
        item.done.wait()
        if item.error is not None:
            raise item.error
        return item.result

    def metrics(self) -> dict[str, int | float | bool | str]:
        """Read without entering the work queue, including during a long learn.

        Wait time is acceptance-to-start, summed over started calls. Execution
        time is start-to-finish, summed over completed calls (including errors).
        Both are host monotonic wall time, not GPU kernel time or utilization.
        Reading metrics neither submits work nor changes activity timestamps.
        """
        with self._metrics_guard:
            return {
                "schema_version": "reap.gpu-actor.metrics.v1",
                "accepting": not self._closed,
                "submitted": self._submitted,
                "started": self._started,
                "completed": self.completed,
                "failed": self._failed,
                "active": self._active,
                "max_active": self.max_active,
                "queued": self._queued,
                "max_queued": self._max_queued,
                "queue_wait_seconds_total": self._queue_wait_seconds,
                "queue_wait_seconds_max": self._max_queue_wait_seconds,
                "execution_seconds_total": self._execution_seconds,
                "execution_seconds_max": self._max_execution_seconds,
                "active_execution_seconds": (monotonic() - self._active_since
                                             if self._active_since is not None else 0.0),
            }

    def close(self) -> None:
        if threading.current_thread() is self._thread:
            raise RuntimeError("GPU actor cannot close from its worker")
        with self._metrics_guard:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        # Concurrent close callers all wait until accepted work is drained.
        self._thread.join()

    def __enter__(self) -> "GpuActor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
