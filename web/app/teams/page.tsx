import Link from "next/link";
import { apiGetOrDefault } from "@/lib/api";
import type { TeamSummary } from "@/lib/types";
import { num, pct } from "@/lib/format";
import { PageHeader, stagger } from "@/components/ui/primitives";

export const revalidate = 600;

export default async function TeamsPage() {
  const teams = await apiGetOrDefault<TeamSummary[]>("/teams", []);
  const byDiv = new Map<string, TeamSummary[]>();
  for (const t of teams) {
    const key = `${t.league ?? "—"} ${t.division ?? ""}`.trim();
    const list = byDiv.get(key) ?? [];
    if (!byDiv.has(key)) byDiv.set(key, list);
    list.push(t);
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Teams" subtitle="Records, ratings, and season simulation odds" />
      {[...byDiv.entries()].map(([div, rows], gi) => (
        <div key={div} style={stagger(gi)} className="stagger-in">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted">{div}</h2>
          <div className="scroll-x rounded-2xl border border-border shadow-card">
            <table className="tabular w-full text-sm">
              <thead className="bg-surface-2 text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="px-3 py-2.5">Team</th>
                  <th className="px-3 py-2.5 text-right">W-L</th>
                  <th className="px-3 py-2.5 text-right">Elo</th>
                  <th className="px-3 py-2.5 text-right">Playoffs</th>
                  <th className="px-3 py-2.5 text-right">Division</th>
                  <th className="px-3 py-2.5 text-right">WS</th>
                  <th className="px-3 py-2.5 text-right">Proj W</th>
                </tr>
              </thead>
              <tbody>
                {rows
                  .sort((a, b) => (b.p_playoffs ?? 0) - (a.p_playoffs ?? 0))
                  .map((t) => (
                    <tr
                      key={t.team_id}
                      className="border-t border-border transition-colors duration-200 hover:bg-surface-2"
                    >
                      <td className="px-3 py-2 font-medium">
                        <Link
                          href={`/teams/${t.abbr}`}
                          className="transition-colors duration-200 hover:text-accent"
                        >
                          {t.abbr} <span className="text-muted">{t.name}</span>
                        </Link>
                      </td>
                      <td className="px-3 py-2 text-right">
                        {t.wins ?? "—"}-{t.losses ?? "—"}
                      </td>
                      <td className="px-3 py-2 text-right">{t.elo ? Math.round(t.elo) : "—"}</td>
                      <td className="px-3 py-2 text-right">{pct(t.p_playoffs)}</td>
                      <td className="px-3 py-2 text-right">{pct(t.p_division)}</td>
                      <td className="px-3 py-2 text-right font-semibold text-accent">
                        {pct(t.p_world_series)}
                      </td>
                      <td className="px-3 py-2 text-right">{num(t.exp_wins, 0)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      {!teams.length ? (
        <p className="text-sm text-muted">No teams ingested yet.</p>
      ) : null}
    </div>
  );
}
