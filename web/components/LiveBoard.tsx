import Link from "next/link";
import { apiGetOrDefault, apiGetOrNull } from "@/lib/api";
import type {
  GameSummary,
  PlayoffsOut,
  PredictionHistoryRow,
  PredictionSummary,
} from "@/lib/types";
import { isoDate, num, pct, shortDate, timeAgo } from "@/lib/format";
import {
  Card,
  CardTitle,
  Empty,
  LiveDot,
  PageHeader,
  StatTile,
  stagger,
} from "@/components/ui/primitives";
import { GameCard } from "@/components/GameCard";
import { AutoRefresh } from "@/components/AutoRefresh";
import { AnimatedNumber } from "@/components/AnimatedNumber";
import { VerdictBadge } from "@/components/VerdictBadge";
import { WSOddsBar } from "@/components/charts/WSOddsBar";

const LIVE_STATUS = /progress|live|in\s|delayed|warmup/i;

export async function LiveBoard() {
  const today = isoDate();
  const [games, summary, graded, playoffs] = await Promise.all([
    apiGetOrDefault<GameSummary[]>(`/games?date=${today}`, [], { revalidate: 0 }),
    apiGetOrNull<PredictionSummary>("/predictions/summary", { revalidate: 0 }),
    apiGetOrDefault<PredictionHistoryRow[]>(
      "/predictions/history?resolved=true&limit=12",
      [],
      { revalidate: 0 },
    ),
    apiGetOrNull<PlayoffsOut>("/playoffs"),
  ]);

  const live = games.filter((g) => !g.is_final && !!g.status && LIVE_STATUS.test(g.status));
  const liveSet = new Set(live);
  const final = games.filter((g) => g.is_final);
  const upcoming = games.filter((g) => !g.is_final && !liveSet.has(g));
  const ordered = [...live, ...upcoming, ...final];
  const gradedToday = final.filter((g) => g.correct != null);
  const rightCount = gradedToday.filter((g) => g.correct).length;

  return (
    <div className="space-y-10">
      <PageHeader
        title="Live board"
        subtitle={
          <span className="inline-flex flex-wrap items-center gap-x-2 gap-y-1">
            {live.length ? <LiveDot /> : null}
            <span>{today} · every game predicted, then graded once it goes final</span>
          </span>
        }
        right={<AutoRefresh intervalMs={60_000} />}
      />

      <Scorecard summary={summary} />

      <section className="space-y-3">
        <CardTitle
          right={
            <Link href="/games" className="text-accent hover:underline">
              Browse all dates →
            </Link>
          }
        >
          Today’s slate
          {gradedToday.length ? (
            <span className="ml-2 font-mono text-fg">
              {rightCount}/{gradedToday.length} correct so far
            </span>
          ) : null}
        </CardTitle>
        {ordered.length ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {ordered.map((g, i) => (
              <GameCard key={g.game_pk} g={g} index={i} />
            ))}
          </div>
        ) : (
          <Empty>No games scheduled today, or the daily pipeline has not run yet.</Empty>
        )}
      </section>

      <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
        <Card>
          <CardTitle
            right={
              <Link href="/predictions" className="text-accent hover:underline">
                Full scorecard →
              </Link>
            }
          >
            Recently graded
          </CardTitle>
          {graded.length ? (
            <ul className="divide-y divide-border text-sm">
              {graded.map((r, i) => (
                <GradedRow key={r.pred_id} r={r} index={i} />
              ))}
            </ul>
          ) : (
            <Empty>No graded predictions yet — check back after tonight’s games.</Empty>
          )}
        </Card>

        <Card>
          <CardTitle
            right={
              <Link href="/playoffs" className="text-accent hover:underline">
                Bracket →
              </Link>
            }
          >
            World Series contenders
          </CardTitle>
          {playoffs?.teams?.length ? (
            <>
              <WSOddsBar teams={playoffs.teams} />
              <p className="mt-2 text-xs text-muted">
                From the latest {playoffs.as_of ? `as-of ${playoffs.as_of} ` : ""}
                100k-season Monte Carlo.
              </p>
            </>
          ) : (
            <Empty>Run a season simulation to populate contender odds.</Empty>
          )}
        </Card>
      </div>
    </div>
  );
}

function Scorecard({ summary }: { summary: PredictionSummary | null }) {
  if (!summary || summary.overall.n === 0) {
    return (
      <Empty>
        No graded predictions yet. Once games go final the resolver fills in
        correct / wrong and this scorecard lights up.
      </Empty>
    );
  }
  const o = summary.overall;
  const w = summary.last_7d;
  const beatsCoin = o.log_loss != null ? o.log_loss < summary.coin_flip_log_loss : null;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <StatTile
        style={stagger(0)}
        label="Accuracy (all-time)"
        value={<AnimatedNumber value={o.accuracy} format={(n) => pct(n, 1)} />}
        hint={`${o.correct}/${o.n} games`}
        tone={o.accuracy != null && o.accuracy >= 0.5 ? "success" : "default"}
      />
      <StatTile
        style={stagger(1)}
        label="Accuracy (last 7d)"
        value={<AnimatedNumber value={w.accuracy} format={(n) => pct(n, 1)} />}
        hint={`${w.correct}/${w.n} games`}
      />
      <StatTile
        style={stagger(2)}
        label="Brier"
        value={<AnimatedNumber value={o.brier} format={(n) => num(n, 3)} />}
        hint="lower is better · 0.25 = coin flip"
      />
      <StatTile
        style={stagger(3)}
        label="Log loss"
        value={<AnimatedNumber value={o.log_loss} format={(n) => num(n, 3)} />}
        hint={`coin flip ${summary.coin_flip_log_loss.toFixed(3)}`}
        tone={beatsCoin === true ? "success" : beatsCoin === false ? "danger" : "default"}
      />
    </div>
  );
}

function GradedRow({ r, index = 0 }: { r: PredictionHistoryRow; index?: number }) {
  const away = r.away_abbr ?? "AWY";
  const home = r.home_abbr ?? "HOM";
  return (
    <li
      style={stagger(index)}
      className="stagger-in flex items-center justify-between gap-3 py-2 transition-colors duration-200 hover:bg-surface-2/60"
    >
      <div className="min-w-0">
        <Link
          href={`/games/${r.game_pk}`}
          className="tabular block truncate font-medium transition-colors duration-200 hover:text-accent"
        >
          {away} {r.away_score ?? "–"} @ {home} {r.home_score ?? "–"}
        </Link>
        <div className="text-xs text-muted">
          {shortDate(r.game_date)} · picked {r.predicted_winner === "home" ? home : away} ·{" "}
          <span className="font-mono">{r.model_id}</span> · {timeAgo(r.created_at)}
        </div>
      </div>
      <VerdictBadge
        v={{
          is_final: true,
          correct: r.correct,
          predicted_winner: r.predicted_winner,
          home_win_prob: r.home_win_prob,
        }}
        homeAbbr={home}
        awayAbbr={away}
      />
    </li>
  );
}
