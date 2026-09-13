# Architecture Decision Records

Short records of choices that fork the architecture and are expensive to reverse.
Format: context → decision → consequences → trade-off.

| # | Decision | Status |
|---|---|---|
| [0001](0001-foundational-decisions.md) | Simulation-first · GitHub Actions · Render/Supabase/Vercel · 2015+ data depth · live deferred to M9 | accepted |
| [0002](0002-vectorised-cohort-simulation.md) | Vectorised cohort simulation, chunked; Numba evaluated and declined | accepted |
| [0003](0003-warehouse-serving-split.md) | Two schemas — `warehouse` (facts) and `serving` (predictions, immutable) | accepted |
| [0004](0004-point-in-time-feature-store.md) | One blessed as-of join (strict `<`) + snapshot hash for every feature | accepted |
| [0005](0005-retrosheet-off-critical-path.md) | Retrosheet off the ingestion critical path — Stats API `allPlays` covers 2015+ | accepted |
| [0006](0006-supabase-session-pooler.md) | Connect to Supabase via the session pooler, not the IPv6-only direct host | accepted |
