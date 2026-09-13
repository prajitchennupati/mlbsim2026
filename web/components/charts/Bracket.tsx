import type { PlayoffsOut, TeamSummary } from "@/lib/types";
import { pct } from "@/lib/format";
import { Badge } from "@/components/ui/primitives";

/**
 * The season Monte Carlo produces per-team probabilities rather than one fixed
 * bracket, so we render the projected field (most likely seeds per league) plus
 * the round-by-round series summaries.
 */
export function Bracket({ data }: { data: PlayoffsOut }) {
  const byLeague = (lg: string) =>
    data.teams
      .filter((t) => (t.league ?? "").includes(lg))
      .sort((a, b) => (b.p_playoffs ?? 0) - (a.p_playoffs ?? 0))
      .slice(0, 6);

  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2">
        <LeagueField title="American League" teams={byLeague("American")} />
        <LeagueField title="National League" teams={byLeague("National")} />
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {data.series.map((s) => (
          <div key={s.round} className="rounded-lg border border-border bg-surface p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold">{roundName(s.round)}</span>
              <Badge>best of {s.best_of}</Badge>
            </div>
            <div className="tabular mt-2 text-sm text-muted">
              avg {s.exp_games.toFixed(1)} games · sweep {pct(s.p_sweep)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function LeagueField({ title, teams }: { title: string; teams: TeamSummary[] }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="mb-2 text-sm font-semibold">{title} — projected field</div>
      <ol className="space-y-1">
        {teams.map((t, i) => (
          <li key={t.team_id} className="tabular flex items-center justify-between text-sm">
            <span>
              <span className="mr-2 inline-block w-4 text-muted">{i + 1}</span>
              {t.abbr}
            </span>
            <span className="text-muted">
              PO {pct(t.p_playoffs)} · WS {pct(t.p_world_series)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function roundName(r: string) {
  const names: Record<string, string> = {
    WC: "Wild Card",
    DS: "Division Series",
    LCS: "Championship",
    WS: "World Series",
  };
  return names[r] ?? r;
}
