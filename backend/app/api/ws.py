"""Week 11: live WebSocket stream.

`/ws/training` is a read-only fan-out of the in-process event bus. It carries no
authority: a client that misses a message refetches the REST endpoint for that
family. The socket exists so a running round, a new audit row or a detection can
appear without waiting for the next poll.

Protocol
--------
On connect the server sends one ``hello`` frame containing the recent history,
then streams ``LiveEvent`` objects exactly as published. An idle connection
receives a lightweight ``heartbeat`` roughly every 20 s so proxies do not drop
it and the client can distinguish "quiet" from "dead".
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.websocket import get_event_bus

log = logging.getLogger(__name__)

router = APIRouter(tags=["live"])

HEARTBEAT_SECONDS = 20.0
HELLO_HISTORY = 50


def _heartbeat() -> dict:
    return {
        "type": "heartbeat",
        "source": "websocket",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {},
    }


@router.websocket("/ws/training")
async def training_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    bus = get_event_bus()
    bus.attach_loop(asyncio.get_running_loop())
    token, queue = bus.subscribe()
    try:
        await websocket.send_json(
            {
                "type": "hello",
                "source": "websocket",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "history": bus.recent(HELLO_HISTORY),
                    "subscribers": bus.subscriber_count,
                    "events_seen": bus.stats()["events_seen"],
                },
            }
        )

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                await websocket.send_json(_heartbeat())
                continue
            await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        # Normal close or a send after the peer went away. Neither is an error.
        pass
    except Exception:  # pragma: no cover - transport failure during shutdown
        log.debug("websocket stream ended", exc_info=True)
    finally:
        bus.unsubscribe(token)


__all__ = ["HEARTBEAT_SECONDS", "HELLO_HISTORY", "router"]
