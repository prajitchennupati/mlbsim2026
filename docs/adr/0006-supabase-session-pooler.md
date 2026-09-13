# ADR 0006 — Connect to Supabase through the session pooler

- **Status:** accepted
- **Date:** 2026-08-31 (decided M10, during the first real-Postgres verification)

## Context

Supabase exposes three connection strings:

| Name | Host | Port | Notes |
|---|---|---|---|
| Direct connection | `db.<ref>.supabase.co` | 5432 | **IPv6-only** unless the paid IPv4 add-on is enabled — no `A` record |
| Session pooler | `aws-0-<region>.pooler.supabase.com` | 5432 | Supavisor in session mode; full session semantics |
| Transaction pooler | same host | 6543 | one Postgres connection per statement; no prepared statements, no `SET`, no advisory locks |

The direct host does not resolve from IPv4-only networks — including GitHub Actions
runners and the dev environment used to build this project.

## Decision

`MLBSIM_DATABASE_URL` uses the **session pooler**:

```
postgresql+psycopg://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require
```

- Username is `postgres.<project-ref>`, not bare `postgres`.
- Session mode (5432), not transaction mode (6543): Alembic migrations, `psycopg`
  prepared statements, and `SET` in `env.py` all need real session semantics.
- The password is percent-encoded in the URL; `cli.py` and `db/migrations/env.py` escape
  `%` → `%%` before handing the URL to Alembic's configparser-based `set_main_option`.

## Consequences

- The same URL works locally, in CI, and on Render.
- `docs/DATABASE.md` and the deployment notes point at the pooler string; the direct
  string is called out as a trap.
- If connection-count pressure ever appears, the transaction pooler is an option for the
  **read-only API only** (with `prepare_threshold=None`), never for the migration or
  pipeline paths.

## Trade-off

One extra network hop (Supavisor) and a region-pinned hostname. Negligible for a batch
pipeline and a cached read API.
