## Week 3 - hospital agents, live

The agent roster and per-site detail pages now read from the FastAPI backend instead of being navigation shells.

### What's real now
- **Class mix per site**, drawn from the actual non-IID partition: Agent A is 93% benign / 1% melanoma, Agent B is 23% melanoma. The skew is shown, not asserted.
- **Partition sizes** come from the capped splitter: 2049 / 1413 / 2050 / 1413 (1.45x spread), logged at seed time.
- **Train / pause / sync** hit the real endpoints and patch the cached roster in place, so a control in one tab is reflected without waiting for a refetch.
- **Empty and unreachable-backend states** say what to start instead of rendering zeroes.

### Honesty fixes
The dashboard delivery progress had a hardcoded \done: true\ for weeks 3-8, which contradicted \CURRENT_IMPLEMENTED_WEEK = 2\ - it claimed six weeks of work that did not exist. Completion is now derived from \CURRENT_IMPLEMENTED_WEEK\, and the stale \SCC\ label in the class hint is gone (HAM10000 has no SCC images).

## Verification
- 50 tests pass (12 new in \	est_agents_api.py\)
- \uff check\ clean
- \
pm run build\ passes
- \GET /api/agents\, \/{id}\, \/{id}/class-distribution\ return 200; train/pause/sync round-trip; unknown ids return 404
