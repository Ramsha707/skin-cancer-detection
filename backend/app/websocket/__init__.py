"""Live event bus surface (Week 11).

Import `get_event_bus` from here. The implementation lives in `bus.py` so the
package keeps a stable public entry point.
"""

from app.websocket.bus import (
    EVENT_TYPES,
    TYPE_AGENT,
    TYPE_AUDIT,
    TYPE_SYSTEM,
    TYPE_TRAINING,
    EventBus,
    get_event_bus,
    set_event_bus,
)

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
