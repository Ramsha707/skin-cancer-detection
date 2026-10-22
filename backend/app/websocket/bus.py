"""Week 11: in-process live event bus.

One bus carries every observable event in the backend to connected WebSocket
clients. It is deliberately small and dependency-free:

* publishers are ordinary threads (the round worker, request handlers), so
  `publish` must be safe from any thread;
* subscribers are asyncio queues owned by WebSocket connections, so delivery
  hops back onto each connection's event loop with `call_soon_threadsafe`;
* a bounded ring buffer keeps the last `history_size` events so a client that
  connects mid-round can render the recent past instead of a blank panel.

The bus is not a message broker. It holds no durable state; the audit ledger is
the durable record. History here exists only to seed a freshly connected client.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

# Event families. The frontend uses these to choose what to invalidate.
TYPE_SYSTEM = "system"
TYPE_AUDIT = "audit"
TYPE_TRAINING = "training"
TYPE_AGENT = "agent"

EVENT_TYPES = (TYPE_SYSTEM, TYPE_AUDIT, TYPE_TRAINING, TYPE_AGENT)

# How many messages a slow subscriber may fall behind before its oldest queued
# event is dropped. Dropping the oldest keeps the stream live rather than
# replaying stale state; the client refetches authoritative REST data anyway.
SUBSCRIBER_QUEUE_SIZE = 128


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventBus:
    """Thread-safe fan-out of live events with a bounded history."""

    def __init__(self, history_size: int = 200) -> None:
        self._lock = threading.RLock()
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._subscribers: dict[int, tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = {}
        self._next_subscriber = 1
        self._sequence = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    # ----------------------------------------------------------- lifecycle --

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the application loop so publishes can cross threads."""
        with self._lock:
            self._loop = loop

    def detach_loop(self) -> None:
        with self._lock:
            self._loop = None

    # ---------------------------------------------------------- subscribers --

    def subscribe(self) -> tuple[int, asyncio.Queue]:
        """Register a subscriber. Returns `(token, queue)`.

        The queue is bound to the running loop when there is one, otherwise to
        the attached application loop. Tests call this from inside the app's
        loop, so the first branch is the normal path.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            with self._lock:
                loop = self._loop
            if loop is None:
                raise RuntimeError(
                    "cannot subscribe outside an event loop before attach_loop()"
                ) from None

        queue: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        with self._lock:
            token = self._next_subscriber
            self._next_subscriber += 1
            self._subscribers[token] = (queue, loop)
        return token, queue

    def unsubscribe(self, token: int) -> None:
        with self._lock:
            self._subscribers.pop(token, None)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # ------------------------------------------------------------ publishing --

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        source: str = "system",
    ) -> dict[str, Any]:
        """Append one event to history and fan it out.

        Returns the event that was published so a caller can log or assert on
        exactly what it emitted.
        """
        with self._lock:
            self._sequence += 1
            event = {
                "id": self._sequence,
                "type": event_type,
                "source": source,
                "timestamp": _utcnow_iso(),
                "payload": payload or {},
            }
            self._history.append(event)
            subscribers = list(self._subscribers.values())

        for queue, loop in subscribers:
            try:
                loop.call_soon_threadsafe(self._offer, queue, event)
            except RuntimeError:  # noqa: PERF203 - only on a closed subscriber loop
                # The subscriber's loop closed between the snapshot and the hop.
                # Dropping the event is correct: that client is gone.
                continue
        return event

    @staticmethod
    def _offer(queue: asyncio.Queue, event: dict[str, Any]) -> None:
        """Put without blocking; drop the oldest event when a client lags."""
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - racy but harmless
                pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - single producer
                pass

    # -------------------------------------------------------------- observers --

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """The last `limit` events, oldest first."""
        if limit < 1:
            return []
        with self._lock:
            items = list(self._history)
        return items[-limit:]

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "subscribers": len(self._subscribers),
                "events_seen": self._sequence,
                "history_size": len(self._history),
            }


_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    global _bus
    with _bus_lock:
        if _bus is None:
            _bus = EventBus()
        return _bus


def set_event_bus(bus: EventBus | None) -> None:
    """Replace the process-wide bus. Test seam."""
    global _bus
    with _bus_lock:
        _bus = bus


__all__ = [
    "EVENT_TYPES",
    "TYPE_AGENT",
    "TYPE_AUDIT",
    "TYPE_SYSTEM",
    "TYPE_TRAINING",
    "EventBus",
    "get_event_bus",
    "set_event_bus",
]
