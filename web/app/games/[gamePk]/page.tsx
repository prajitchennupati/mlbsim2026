import { notFound } from "next/navigation";
import { apiGetOrNull } from "@/lib/api";
import type {
  ExplanationOut,
  GameDetail,
  GamePlayers,
  InningTable,
  SimulationOut,
} from "@/lib/types";
import { clockTime, num, pct, shortDate } from "@/lib/format";
import { Card, CardTitle, ProbBar, StatTile, stagger } from "@/components/ui/primitives";
import { VerdictBadge } from "@/components/VerdictBadge";
import { FactorList } from "@/components/FactorList";
import { RunDistBar } from "@/components/charts/RunDistBar";
import { ScoreHeatmap } from "@/components/charts/ScoreHeatmap";
import { InningRunsTable } from "@/components/charts/InningRunsTable";
import { cn } from "@/lib/cn";

export const revalidate = 120;

export default async function GamePage({
  params,
}: {
  params: Promise<{ gamePk: string }>;
}) {
  const { gamePk } = await params;
  const game = await apiGetOrNull<GameDetail>(`/games/${gamePk}`, { revalidate: 60 });
  if (!game) notFound();

  const [players, innings, sim, explain] = await Promise.all([
    apiGetOrNull<GamePlayers>(`/games/${gamePk}/players`),
    apiGetOrNull<InningTable>(`/games/${gamePk}/innings`),
    apiGetOrNull<SimulationOut>(`/games/${gamePk}/simulation`),
    game.pred_id
      ? apiGetOrNull<ExplanationOut>(`/predictions/${game.pred_id}/explanation`)
      : Promise.resolve(null),
  ]);

  const home = game.home_abbr ?? String(game.home_team_id);
  const away = game.away_abbr ?? String(game.away_team_id);
  const hp = game.home_win_prob ?? 0.5;
  const final = game.is_final;
  const homeWon = final && (game.home_score ?? 0) > (game.away_score ?? 0);

  return (
    <div className="space-y-6">
      <div className="animate-fade-in-up">
        <div className="flex items-center gap-2 text-sm text-muted">
          <span>
            {shortDate(game.game_date)} ·{" "}
            {final ? "Final" : game.status || clockTime(game.start_time_utc) || "Scheduled"}
          </span>
          {game.model_id ? <span className="font-mono">· {game.model_id}</span> : null}
        </div>
        <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <h1 className="flex items-baseline gap-3 text-3xl font-bold tracking-tight">
            <span
              className={cn(
                "text-away transition-colors duration-500",
                final && !homeWon ? "" : final && "text-muted",
              )}
            >
              {away}
              {final ? <span className="ml-2">{game.away_score}</span> : null}
            </span>
            <span className="text-muted">@</span>
            <span
              className={cn(
                "text-home transition-colors duration-500",
                final && homeWon ? "" : final && "text-muted",
              )}
            >
              {home}
              {final ? <span className="ml-2">{game.home_score}</span> : null}
            </span>
          </h1>
          <VerdictBadge v={game} homeAbbr={home} awayAbbr={away} size="lg" />
        </div>
      </div>

      {final && game.correct != null ? (
        <ResultBanner game={game} home={home} away={away} />
      ) : null}

      <Card style={stagger(0)}>
        <div className="flex items-center justify-between text-sm font-medium">
          <span className="text-away">
            {away} {pct(1 - hp)}
          </span>
          <span className="text-home">
            {home} {pct(hp)}
          </span>
        </div>
        <div className="mt-2">
          <ProbBar home={hp} away={1 - hp} />
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile
            label="Proj score"
            value={`${num(game.exp_away_runs)}–${num(game.exp_home_runs)}`}
            hint={final ? `actual ${game.away_score}–${game.home_score}` : undefined}
          />
          <StatTile label="Extra innings" value={pct(game.p_extra_innings)} />
          <StatTile label="One-run game" value={pct(game.p_one_run_game)} />
          <StatTile
            label="Shutout A / H"
            value={`${pct(game.p_shutout_away)} / ${pct(game.p_shutout_home)}`}
          />
        </div>
        {explain?.explanation ? (
          <p className="mt-4 border-t border-border pt-3 text-sm text-muted">
            {explain.explanation}
          </p>
        ) : null}
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card style={stagger(1)}>
          <CardTitle>Why this prediction</CardTitle>
          <FactorList
            factors={game.factors?.top_factors ?? explain?.top_factors ?? []}
            homeAbbr={home}
            awayAbbr={away}
          />
        </Card>
        <Card style={stagger(2)}>
          <CardTitle>Most likely final scores</CardTitle>
          <ScoreHeatmap scores={game.most_likely_scores} homeAbbr={home} awayAbbr={away} />
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card style={stagger(1)}>
          <CardTitle>{away} run distribution</CardTitle>
          <RunDistBar dist={game.away_score_dist} label={`${away} runs`} />
        </Card>
        <Card style={stagger(2)}>
          <CardTitle>{home} run distribution</CardTitle>
          <RunDistBar dist={game.home_score_dist} label={`${home} runs`} />
        </Card>
      </div>

      {innings && innings.rows.length ? (
        <Card>
          <CardTitle>Runs by inning (simulated)</CardTitle>
          <InningRunsTable
            rows={innings.rows}
            homeAbbr={home}
            awayAbbr={away}
            leadAfter={innings.p_home_lead_after}
          />
        </Card>
      ) : null}

      {players && (players.batters.length || players.pitchers.length) ? (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card style={stagger(1)}>
            <CardTitle>Batter projections</CardTitle>
            <BatterTable rows={players.batters} />
          </Card>
          <Card style={stagger(2)}>
            <CardTitle>Starter projections</CardTitle>
            <PitcherTable rows={players.pitchers} />
          </Card>
        </div>
      ) : null}

      {sim ? (
        <p className="text-xs text-muted">
          Simulation: {sim.n_sims.toLocaleString()} games · engine {sim.engine_version}
          {sim.runtime_ms ? ` · ${sim.runtime_ms} ms` : ""}
        </p>
      ) : null}
    </div>
  );
}

function ResultBanner({
  game,
  home,
  away,
}: {
  game: GameDetail;
  home: string;
  away: string;
}) {
  const ok = game.correct === true;
  const pickedAbbr = game.predicted_winner === "home" ? home : away;
  const wonAbbr = game.actual_winner === "home" ? home : away;
  const conf =
    game.predicted_winner === "home"
      ? game.home_win_prob ?? 0.5
      : 1 - (game.home_win_prob ?? 0.5);
  return (
    <div
      className={cn(
        "animate-scale-in rounded-2xl border p-4 text-sm shadow-card",
        ok ? "border-success/30 bg-success/10" : "border-danger/30 bg-danger/10",
      )}
    >
      <span className={cn("font-semibold", ok ? "text-success" : "text-danger")}>
        {ok ? "✓ Correct" : "✗ Miss"}
      </span>{" "}
      — the model picked <strong>{pickedAbbr}</strong> at {pct(conf)}; {wonAbbr} won{" "}
      {game.away_score}–{game.home_score}.
      {game.brier != null ? (
        <span className="text-muted"> Brier {game.brier.toFixed(3)}.</span>
      ) : null}
    </div>
  );
}

function BatterTable({ rows }: { rows: GamePlayers["batters"] }) {
  if (!rows.length) return <p className="text-sm text-muted">No batter projections.</p>;
  return (
    <div className="scroll-x">
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wider text-muted">
            <th className="py-1">Player</th>
            <th className="px-2 py-1 text-right">PA</th>
            <th className="px-2 py-1 text-right">H</th>
            <th className="px-2 py-1 text-right">HR</th>
            <th className="px-2 py-1 text-right">P(1+H)</th>
            <th className="px-2 py-1 text-right">P(HR)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.player_id} className="border-t border-border transition-colors duration-200 hover:bg-surface-2">
              <td className="py-1.5">#{r.player_id}</td>
              <td className="px-2 py-1.5 text-right">{num(r.proj.pa, 1)}</td>
              <td className="px-2 py-1.5 text-right">{num(r.proj.h, 2)}</td>
              <td className="px-2 py-1.5 text-right">{num(r.proj.hr, 2)}</td>
              <td className="px-2 py-1.5 text-right">{pct(r.prob.p_1plus_h)}</td>
              <td className="px-2 py-1.5 text-right">{pct(r.prob.p_hr)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PitcherTable({ rows }: { rows: GamePlayers["pitchers"] }) {
  if (!rows.length) return <p className="text-sm text-muted">No starter projections.</p>;
  return (
    <div className="scroll-x">
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wider text-muted">
            <th className="py-1">Pitcher</th>
            <th className="px-2 py-1 text-right">IP</th>
            <th className="px-2 py-1 text-right">K</th>
            <th className="px-2 py-1 text-right">P(6+K)</th>
            <th className="px-2 py-1 text-right">P(6+IP)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.player_id} className="border-t border-border transition-colors duration-200 hover:bg-surface-2">
              <td className="py-1.5">#{r.player_id}</td>
              <td className="px-2 py-1.5 text-right">{num(r.proj.mean_ip, 1)}</td>
              <td className="px-2 py-1.5 text-right">{num(r.proj.mean_k, 1)}</td>
              <td className="px-2 py-1.5 text-right">{pct(r.prob.p_6plus_k)}</td>
              <td className="px-2 py-1.5 text-right">{pct(r.prob.p_6plus_ip)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
