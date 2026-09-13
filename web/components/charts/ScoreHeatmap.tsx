"use client";

import { cn } from "@/lib/cn";

type Cell = { home: number; away: number; p: number };

export function ScoreHeatmap({
  scores,
  homeAbbr = "H",
  awayAbbr = "A",
  size = 9,
}: {
  scores: Cell[] | null | undefined;
  homeAbbr?: string;
  awayAbbr?: string;
  size?: number;
}) {
  if (!scores?.length) {
    return <div className="py-6 text-center text-sm text-muted">no simulated scores</div>;
  }
  const grid = new Map<string, number>();
  let peak = 0;
  for (const s of scores) {
    if (s.home > size || s.away > size) continue;
    grid.set(`${s.home}-${s.away}`, s.p);
    peak = Math.max(peak, s.p);
  }
  const rows = Array.from({ length: size + 1 }, (_, i) => i);

  return (
    <div className="overflow-x-auto">
      <table className="tabular border-separate border-spacing-[2px] text-[10px]">
        <thead>
          <tr>
            <th className="w-6" />
            <th className="pb-1 text-left text-muted" colSpan={size + 1}>
              {awayAbbr} runs →
            </th>
          </tr>
          <tr>
            <th className="text-muted">{homeAbbr}↓</th>
            {rows.map((a) => (
              <th key={a} className="w-6 text-center text-muted">
                {a}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((h) => (
            <tr key={h}>
              <td className="pr-1 text-right text-muted">{h}</td>
              {rows.map((a) => {
                const p = grid.get(`${h}-${a}`) ?? 0;
                const intensity = peak ? p / peak : 0;
                return (
                  <td
                    key={a}
                    title={`${homeAbbr} ${h}, ${awayAbbr} ${a}: ${(p * 100).toFixed(1)}%`}
                    className={cn(
                      "h-6 w-6 rounded text-center",
                      h > a ? "text-home" : h < a ? "text-away" : "text-muted",
                    )}
                    style={{
                      background:
                        intensity > 0.02
                          ? `rgb(var(--accent) / ${(0.12 + intensity * 0.7).toFixed(2)})`
                          : "rgb(var(--border) / 0.35)",
                    }}
                  >
                    {p > 0.012 ? Math.round(p * 100) : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
