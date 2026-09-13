import { notFound } from "next/navigation";
import { apiGet, apiGetOrNull } from "@/lib/api";
import type { GameSummary, TeamSummary } from "@/lib/types";
import { num, pct, shortDate } from "@/lib/format";
import { Card, CardTitle, PageHeader, StatTile } from "@/components/ui/primitives";
import { GameCard } from "@/components/GameCard";

export const revalidate = 600;

export default async function TeamPage({ params }: { params: Promise<{ abbr: string }> }) {
  const { abbr } = await params;
  const team = await apiGetOrNull<TeamSummary>(`/teams/${abbr}`);
  if (!team) notFound();
  const schedule = await apiGet<GameSummary[]>(`/teams/${abbr}/schedule`).catch(
    () => [] as GameSummary[],
  );

  const today = new Date().toISOString().slice(0, 10);
  const upcoming = schedule.filter((g) => (g.game_date ?? "") >= today).slice(0, 9);
  const recent = schedule
    .filter((g) => (g.game_date ?? "") < today)
    .slice(-6)
    .reverse();

  return (
    <div className="space-y-8">
      <PageHeader
        title={`${team.abbr} — ${team.name ?? ""}`}
        subtitle={`${team.league ?? ""} ${team.division ?? ""}`.trim()}
      />

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatTile label="Record" value={`${team.wins ?? "—"}-${team.losses ?? "—"}`} />
        <StatTile label="Elo" value={team.elo ? Math.round(team.elo) : "—"} />
        <StatTile label="Proj wins" value={num(team.exp_wins, 0)} />
        <StatTile label="Playoffs" value={pct(team.p_playoffs)} />
        <StatTile label="Division" value={pct(team.p_division)} />
        <StatTile label="World Series" value={pct(team.p_world_series)} />
      </div>

      {upcoming.length ? (
        <section>
          <CardTitle>Upcoming</CardTitle>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {upcoming.map((g) => (
              <GameCard key={g.game_pk} g={g} />
            ))}
          </div>
        </section>
      ) : null}

      {recent.length ? (
        <Card>
          <CardTitle>Recent results</CardTitle>
          <ul className="tabular divide-y divide-border text-sm">
            {recent.map((g) => {
              const isHome = g.home_team_id === team.team_id;
              const us = isHome ? g.home_score : g.away_score;
              const them = isHome ? g.away_score : g.home_score;
              const opp = isHome ? g.away_abbr : g.home_abbr;
              const won = us != null && them != null && us > them;
              return (
                <li key={g.game_pk} className="flex items-center justify-between py-2">
                  <span>
                    {shortDate(g.game_date)} {isHome ? "vs" : "@"} {opp}
                  </span>
                  <span className={won ? "text-home" : "text-away"}>
                    {won ? "W" : "L"} {us}-{them}
                  </span>
                </li>
              );
            })}
          </ul>
        </Card>
      ) : null}
    </div>
  );
}
