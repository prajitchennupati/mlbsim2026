"use client";

import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { distToArray } from "@/lib/format";

export function RunDistBar({
  dist,
  label = "runs",
  max = 12,
}: {
  dist: Record<string, number> | null | undefined;
  label?: string;
  max?: number;
}) {
  const data = distToArray(dist, max);
  if (!data.length) {
    return <div className="py-6 text-center text-sm text-muted">no distribution</div>;
  }
  const peak = data.reduce((a, b) => (b.p > a.p ? b : a));

  return (
    <ResponsiveContainer width="100%" height={140}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: -28 }}>
        <XAxis
          dataKey="value"
          tick={{ fontSize: 11, fill: "rgb(var(--muted))" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis hide />
        <Tooltip
          formatter={(v) => [`${(Number(v) * 100).toFixed(1)}%`, "prob"]}
          labelFormatter={(l) => `${l} ${label}`}
          contentStyle={{
            background: "rgb(var(--surface))",
            border: "1px solid rgb(var(--border))",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Bar dataKey="p" radius={[3, 3, 0, 0]}>
          {data.map((d) => (
            <Cell
              key={d.value}
              fill={d.value === peak.value ? "rgb(var(--accent))" : "rgb(var(--border))"}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
