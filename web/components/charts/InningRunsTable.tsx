import type { InningRow } from "@/lib/types";
import { num, pct } from "@/lib/format";

export function InningRunsTable({
  rows,
  homeAbbr = "Home",
  awayAbbr = "Away",
  leadAfter = [],
}: {
  rows: InningRow[];
  homeAbbr?: string;
  awayAbbr?: string;
  leadAfter?: number[];
}) {
  if (!rows.length) {
    return <div className="py-6 text-center text-sm text-muted">no inning simulation</div>;
  }
  const innings = Array.from(new Set(rows.map((r) => r.inning))).sort((a, b) => a - b);
  const by = (side: string, inn: number) => rows.find((r) => r.side === side && r.inning === inn);

  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wide text-muted">
            <th className="py-1 pr-3">Inning</th>
            {innings.map((i) => (
              <th key={i} className="px-2 py-1 text-center">
                {i > 9 ? "X" : i}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr className="border-t border-border">
            <td className="py-1.5 pr-3 font-medium text-away">{awayAbbr} exp R</td>
            {innings.map((i) => (
              <td key={i} className="px-2 py-1.5 text-center">
                {num(by("away", i)?.exp_runs, 2)}
              </td>
            ))}
          </tr>
          <tr className="border-t border-border">
            <td className="py-1.5 pr-3 font-medium text-home">{homeAbbr} exp R</td>
            {innings.map((i) => (
              <td key={i} className="px-2 py-1.5 text-center">
                {num(by("home", i)?.exp_runs, 2)}
              </td>
            ))}
          </tr>
          {leadAfter.length ? (
            <tr className="border-t border-border text-muted">
              <td className="py-1.5 pr-3">P({homeAbbr} lead)</td>
              {innings.map((i, idx) => (
                <td key={i} className="px-2 py-1.5 text-center">
                  {leadAfter[idx] !== undefined ? pct(leadAfter[idx]) : "—"}
                </td>
              ))}
            </tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}
