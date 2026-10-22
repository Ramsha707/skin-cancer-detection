import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";

import { trainingSocketUrl } from "@/services/api";
import type { LiveEvent, LiveEventType } from "@/types";

/**
 * Week 11: one WebSocket per console session.
 *
 * The socket is a *notification* channel, not a data source. Each event only
 * invalidates the REST queries that own its family, so the page still renders
 * authoritative DB state. A dropped message therefore costs a late refresh,
 * never a wrong number.
 */

export type LiveStatus = "connecting" | "live" | "reconnecting" | "offline";

interface LiveEventsValue {
  status: LiveStatus;
  events: LiveEvent[];
  lastEventAt: string | null;
  lastHeartbeatAt: string | null;
  reconnects: number;
  error: string | null;
  clear: () => void;
}

const MAX_EVENTS = 250;

const LiveEventsContext = createContext<LiveEventsValue | null>(null);

const INVALIDATIONS: Record<LiveEventType, string[][]> = {
  audit: [["audit"], ["dashboard"], ["privacy"], ["federated"], ["rounds"]],
  training: [["federated"], ["rounds"], ["dashboard"], ["agents"]],
  agent: [["federated"], ["agents"]],
  system: [["health"], ["integration"], ["system-events"], ["demo"], ["dashboard"]],
};

function isLiveEvent(value: unknown): value is LiveEvent {
  if (!value || typeof value !== "object") return false;
  const e = value as Partial<LiveEvent>;
  return (
    typeof e.id === "number" &&
    typeof e.type === "string" &&
    typeof e.source === "string" &&
    typeof e.timestamp === "string"
  );
}

function appendCapped(current: LiveEvent[], incoming: LiveEvent[]): LiveEvent[] {
  const seen = new Set(current.map((e) => e.id));
  const merged = [...current];
  for (const event of incoming) {
    if (seen.has(event.id)) continue;
    seen.add(event.id);
    merged.push(event);
  }
  return merged.slice(-MAX_EVENTS);
}

function useInvalidation(queryClient: QueryClient) {
  const timers = useRef<Partial<Record<LiveEventType, number>>>({});

  useEffect(
    () => () => {
      Object.values(timers.current).forEach((id) => window.clearTimeout(id));
      timers.current = {};
    },
    [],
  );

  return useCallback(
    (type: LiveEventType) => {
      if (timers.current[type] !== undefined) return;
      // A round emits ~20 audit rows in under a second. One refresh per family
      // after the burst is both cheaper and less flickery than 20.
      timers.current[type] = window.setTimeout(() => {
        delete timers.current[type];
        for (const key of INVALIDATIONS[type]) {
          void queryClient.invalidateQueries({ queryKey: key });
        }
      }, 350);
    },
    [queryClient],
  );
}

export function LiveEventsProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const scheduleInvalidation = useInvalidation(queryClient);

  const [status, setStatus] = useState<LiveStatus>("connecting");
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [lastEventAt, setLastEventAt] = useState<string | null>(null);
  const [lastHeartbeatAt, setLastHeartbeatAt] = useState<string | null>(null);
  const [reconnects, setReconnects] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | null = null;
    let retry = 0;
    let retryTimer: number | undefined;

    const connect = () => {
      if (disposed) return;
      setStatus(retry === 0 ? "connecting" : "reconnecting");
      setError(null);

      socket = new WebSocket(trainingSocketUrl());

      socket.onopen = () => {
        if (disposed) return;
        retry = 0;
        setStatus("live");
        setReconnects((n) => n + 1);
      };

      socket.onmessage = (message) => {
        if (disposed) return;
        let frame: unknown;
        try {
          frame = JSON.parse(message.data as string);
        } catch {
          return;
        }
        if (!frame || typeof frame !== "object") return;
        const payload = (frame as { payload?: unknown }).payload;
        const type = (frame as { type?: unknown }).type;

        if (type === "hello") {
          const history =
            payload && typeof payload === "object" && Array.isArray((payload as { history?: unknown }).history)
              ? ((payload as { history: unknown[] }).history.filter(isLiveEvent) as LiveEvent[])
              : [];
          setEvents((current) => appendCapped(current, history));
          setStatus("live");
          return;
        }

        if (type === "heartbeat") {
          const timestamp = (frame as { timestamp?: unknown }).timestamp;
          if (typeof timestamp === "string") setLastHeartbeatAt(timestamp);
          return;
        }

        if (type === "system" || type === "audit" || type === "training" || type === "agent") {
          if (!isLiveEvent(frame)) return;
          setEvents((current) => appendCapped(current, [frame]));
          setLastEventAt(frame.timestamp);
          scheduleInvalidation(frame.type);
        }
      };

      socket.onerror = () => {
        setError("Live stream interrupted; retrying.");
      };

      socket.onclose = () => {
        if (disposed) return;
        setStatus("reconnecting");
        retry += 1;
        const delay = Math.min(1000 * 2 ** Math.min(retry - 1, 4), 10_000);
        retryTimer = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      disposed = true;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      socket?.close();
    };
  }, [scheduleInvalidation]);

  const clear = useCallback(() => {
    setEvents([]);
    setLastEventAt(null);
  }, []);

  const value = useMemo<LiveEventsValue>(
    () => ({
      status,
      events,
      lastEventAt,
      lastHeartbeatAt,
      reconnects,
      error,
      clear,
    }),
    [status, events, lastEventAt, lastHeartbeatAt, reconnects, error, clear],
  );

  return <LiveEventsContext.Provider value={value}>{children}</LiveEventsContext.Provider>;
}

export function useLiveEvents(): LiveEventsValue {
  const value = useContext(LiveEventsContext);
  if (!value) throw new Error("useLiveEvents must be used inside <LiveEventsProvider>");
  return value;
}
