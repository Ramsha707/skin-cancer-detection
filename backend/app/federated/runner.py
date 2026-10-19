"""Week 7: background round execution.

Rounds run on one daemon worker thread so `POST /api/federated/start` returns
immediately and the dashboard can poll `/state`. Pause takes effect at the next
round boundary: a round in flight always finishes, because aborting mid-FedAvg
would leave agents holding different global versions.

The worker is deliberately free of FastAPI and SQLAlchemy imports; it opens one
session per round through an injectable factory, which is also the seam tests
use to point it at a throwaway database.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

# Statuses a round queue can be in. Mirrors the frontend's TrainingStatus.
STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"


class RoundRunner:
    """Single-worker queue for federated rounds.

    `execute_round`, when supplied, replaces the default "open a session and run
    one real round" body. Tests inject a counter or a sleeper here to exercise
    queue and pause semantics deterministically.
    """

    def __init__(self, execute_round: Callable[[], None] | None = None) -> None:
        self._lock = threading.RLock()
        self._queued = 0
        self._pause = False
        self._status = STATUS_IDLE
        self._aggregation_status = "idle"
        self._error: str | None = None
        self._worker: threading.Thread | None = None
        self._agent_live: dict[str, dict] = {}
        self._execute_round = execute_round
        # Set by the API (or a test) to the session factory used per round.
        self.session_factory = None

    # ------------------------------------------------------------- observers --

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def is_active(self) -> bool:
        """True while a run is in progress or paused."""
        with self._lock:
            return self._status in (STATUS_RUNNING, STATUS_PAUSED)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "aggregation_status": self._aggregation_status,
                "queued": self._queued,
                "error": self._error,
            }

    def agent_live(self) -> dict[str, dict]:
        with self._lock:
            return {agent_id: dict(fields) for agent_id, fields in self._agent_live.items()}

    def note_agent(self, agent_id: str, **fields) -> None:
        """Record live per-agent progress for `/state`. Metrics only."""
        with self._lock:
            self._agent_live.setdefault(agent_id, {}).update(fields)

    # ------------------------------------------------------------ control --

    def request_rounds(self, n: int) -> int:
        """Queue `n` more rounds (resuming from pause). Returns queue depth."""
        if n < 1:
            raise ValueError("request_rounds requires at least one round")
        with self._lock:
            self._queued += n
            self._pause = False
            self._status = STATUS_RUNNING
            self._error = None
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._work, name="federated-rounds", daemon=True
                )
                self._worker.start()
            return self._queued

    def pause(self) -> None:
        """Drop pending rounds; the round in flight finishes at the boundary."""
        with self._lock:
            self._pause = True
            self._queued = 0
            if self._status == STATUS_RUNNING:
                self._status = STATUS_PAUSED

    def reset(self, *, timeout: float = 120.0) -> None:
        """Stop the worker and clear runtime state.

        Database history and the audit ledger are untouched; only in-memory
        engine state and live agent progress are dropped.
        """
        with self._lock:
            self._pause = True
            self._queued = 0
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
        with self._lock:
            self._worker = None
            self._pause = False
            self._status = STATUS_IDLE
            self._aggregation_status = "idle"
            self._error = None
            self._agent_live.clear()

        from app.federated.engine import get_engine

        get_engine().reset()

    # ------------------------------------------------------------- worker --

    def _work(self) -> None:
        while True:
            with self._lock:
                if self._pause or self._queued <= 0:
                    if self._status == STATUS_RUNNING:
                        self._status = STATUS_PAUSED if self._pause else STATUS_COMPLETED
                    self._aggregation_status = "idle"
                    return
                self._queued -= 1
                self._aggregation_status = "running"
            try:
                self._run_one()
            except Exception as exc:
                log.exception("federated round failed")
                with self._lock:
                    self._status = STATUS_ERROR
                    self._error = str(exc)
                    self._queued = 0
                    self._aggregation_status = "idle"
                return

    def _run_one(self) -> None:
        if self._execute_round is not None:
            self._execute_round()
            return

        from app.database.base import SessionLocal
        from app.services import training as training_service

        factory = self.session_factory or SessionLocal
        db = factory()
        try:
            training_service.run_round(db, runner=self)
        finally:
            db.close()


_runner: RoundRunner | None = None


def get_runner() -> RoundRunner:
    global _runner
    if _runner is None:
        _runner = RoundRunner()
    return _runner


def set_runner(runner: RoundRunner | None) -> None:
    """Test seam."""
    global _runner
    _runner = runner


__all__ = [
    "STATUS_COMPLETED",
    "STATUS_ERROR",
    "STATUS_IDLE",
    "STATUS_PAUSED",
    "STATUS_RUNNING",
    "RoundRunner",
    "get_runner",
    "set_runner",
]
