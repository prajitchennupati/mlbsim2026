import Link from "next/link";
import { apiGetOrDefault } from "@/lib/api";
import type { StandingRow } from "@/lib/types";
import { PageHeader } from "@/components/ui/primitives";

export const revalidate = 600;

export default async function StandingsPage() {
  const rows = await apiGetOrDefault<StandingRow[]>("/standings", []);
  const groups = new Map<string, StandingRow[]>();
  for (const r of rows) {
    const key = `${r.league ?? "—"} ${r.division ?? ""}`.trim();
    const list = groups.get(key) ?? [];
    if (!groups.has(key)) groups.set(key, list);
    list.push(r);
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Standings" subtitle="Actual records to date" />
      {[...groups.entries()].map(([div, list]) => (
        <div key={div}>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted">{div}</h2>
          <div className="overflow-x-auto rounded-xl border border-border">
            <table className="tabular w-full text-sm">
              <thead className="bg-surface text-left text-xs uppercase text-muted">
                <tr>
                  <th className="px-3 py-2">Team</th>
                  <th className="px-3 py-2 text-right">W</th>
                  <th className="px-3 py-2 text-right">L</th>
                  <th className="px-3 py-2 text-right">Pct</th>
                  <th className="px-3 py-2 text-right">GB</th>
                  <th className="px-3 py-2 text-right">Diff</th>
                </tr>
              </thead>
              <tbody>
                {list
                  .sort((a, b) => b.pct - a.pct)
                  .map((r, i) => {
                    const leader = list[0];
                    const gb = ((leader.wins - r.wins + (r.losses - leader.losses)) / 2).toFixed(1);
                    return (
                      <tr key={r.team_id} className="border-t border-border">
                        <td className="px-3 py-2 font-medium">
                          <Link href={`/teams/${r.abbr}`} className="hover:text-accent">
                            {r.abbr}
                          </Link>
                        </td>
                        <td className="px-3 py-2 text-right">{r.wins}</td>
                        <td className="px-3 py-2 text-right">{r.losses}</td>
                        <td className="px-3 py-2 text-right">{r.pct.toFixed(3)}</td>
                        <td className="px-3 py-2 text-right">{i === 0 ? "—" : gb}</td>
                        <td
                          className={`px-3 py-2 text-right ${
                            r.run_diff > 0 ? "text-home" : r.run_diff < 0 ? "text-away" : ""
                          }`}
                        >
                          {r.run_diff > 0 ? "+" : ""}
                          {r.run_diff}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      {!rows.length ? <p className="text-sm text-muted">No games logged yet.</p> : null}
    </div>
  );
}
