# Data Sources & Requirements

## 1. Sources

| Source | Access | Coverage | Pull cadence | Feeds |
|---|---|---|---|---|
| **MLB Stats API** (`statsapi.mlb.com`, GUMBO live feed) | free JSON, undocumented for bulk | schedules, rosters, probable pitchers, **full play-by-play (`allPlays`) 2015–present**, boxscores, standings, transactions | schedule + rosters daily; live feed every 15–30 s during games | primary 2026 pipeline; **all `plate_appearances` / `pitches` / `lineups` for 2015+**; live win probability (M9); `game_probables`, `transactions`, `standings` |
| **Baseball Savant / Statcast** (CSV export endpoint, date-chunked) | free CSV, rate-limited | pitch-level tracking, batted-ball, expected stats (xBA/xwOBA/barrel); 2015–present | daily incremental (previous day) | enrichment of `pitches` / `batted_balls` with tracking + expected stats; hitter/pitcher quality metrics; player embeddings |
| **Retrosheet** event files | free download, ~1-year release lag, attribution required | full play-by-play 1918–2024 | **optional — only if history is extended before 2015** | pre-2015 `plate_appearances`; longer baserunning-matrix and Elo history |
| **FanGraphs** (via `pybaseball`) | free, rate-limited | wOBA, wRC+, FIP/xFIP/SIERA, park factors, Steamer & ZiPS projections | weekly | derived rate features; external projection benchmark (not a component) |
| **Baseball-Reference** (via `pybaseball`) | free, rate-limited | historical splits, park factors | monthly | cross-validation of park factors and splits |
| **Chadwick Bureau register** (`chadwick-bureau/register` on GitHub) | free CSV, CC-0 | player ID crosswalk MLBAM ↔ Retrosheet ↔ FanGraphs ↔ Baseball-Reference | monthly | `players` ID reconciliation (`crosswalk.py`) |
| **Open-Meteo** | free API, no key | historical + forecast weather by lat/lon | forecast pull daily for upcoming games; historical backfill once | `game_weather` (forecast); context features |
| Vegas / market lines | ToS-dependent | closing moneyline / total | **optional, disabled by default** | evaluation baseline **only** — never a model input (would leak other models' information and raises licensing questions) |

## 2. Backfill scope (locked — ADR 0001)

| Data | From |
|---|---|
| `plate_appearances`, `pitches`, `batted_balls`, Statcast-derived features | **2015** |
| `games`, `team_game_logs`, `batting_game_logs`, `pitching_game_logs`, Elo | **2010** |
| `parks`, `players`, `seasons` dimensions | as far back as needed for the above |

Rationale: Statcast starts 2015, so a single-regime event model wants 2015+. Games back to
2010 give Elo and the win/loss baselines a longer warm-up without forcing the event model
to straddle a pre-Statcast feature regime. Extending PA data back to ~2000 (no Statcast) is
a possible later addition.

## 3. Volume

- ~2,430 games/season × ~76 PA ≈ 185k PA rows/season → 2015–2025 ≈ 2.0M PA rows.
- Statcast ≈ 700k pitches/season → kept as Parquet in object storage; only modelled
  columns land in `pitches` / `batted_balls`.
- Comfortable on Supabase low tiers; large raw pulls never touch Postgres.

## 4. Ingestion contract

1. Every external response is written verbatim to the **raw landing zone** (local cache +
   Parquet in object storage) keyed by request URL + fetch timestamp, before any parsing.
2. Parsed rows pass a **pandera** schema (types, ranges, nullability, referential keys)
   before load; failures quarantine the batch and alert, they do not partially load.
3. Loads are **idempotent upserts** on natural keys — re-running a day is safe.
4. A **nightly reconcile** re-pulls the full schedule for the trailing 3 days to catch
   corrections (rain-outs, stat revisions, roster moves) missed by the incremental pull.
5. Point-in-time facts (`game_probables`, `lineups`, `standings`, `transactions`) are
   append-only with `source_ts`; the feature layer only ever reads them as-of.

## 5. Terms of use & attribution

This is a non-commercial research / portfolio project.

- **Retrosheet** — the project displays the required notice: *"The information used here was
  obtained free of charge from and is copyrighted by Retrosheet. Interested parties may
  contact Retrosheet at www.retrosheet.org."*
- **MLB Stats API** — used for personal, non-commercial analysis; no MLB marks are used to
  imply endorsement; no redistribution of bulk raw feeds.
- **Baseball Savant / FanGraphs / Baseball-Reference** — accessed at polite rates via
  `pybaseball`; figures are attributed on the methodology page.
- **Chadwick Bureau register** — CC-0.
- **Open-Meteo** — free for non-commercial use, attributed.
