"use client";

import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TeamSummary } from "@/lib/types";

export function WSOddsBar({ teams, limit = 12 }: { teams: TeamSummary[]; limit?: number }) {
  const data = [...teams]
    .filter((t) => (t.p_world_series ?? 0) > 0)
    .sort((a, b) => (b.p_world_series ?? 0) - (a.p_world_series ?? 0))
    .slice(0, limit)
    .map((t) => ({
      abbr: t.abbr ?? String(t.team_id),
      p: (t.p_world_series ?? 0) * 100,
      playoffs: (t.p_playoffs ?? 0) * 100,
    }));

  if (!data.length) {
    return <div className="py-6 text-center text-sm text-muted">no season simulation yet</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 26)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 40, bottom: 4, left: 8 }}>
        <XAxis type="number" hide />
        <YAxis
          type="category"
          dataKey="abbr"
          width={44}
          tick={{ fontSize: 12, fill: "rgb(var(--fg))" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          formatter={(v, k) => [`${Number(v).toFixed(1)}%`, k === "p" ? "World Series" : "Playoffs"]}
          contentStyle={{
            background: "rgb(var(--surface))",
            border: "1px solid rgb(var(--border))",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Bar dataKey="p" radius={[0, 4, 4, 0]}>
          {data.map((d) => (
            <Cell key={d.abbr} fill="rgb(var(--accent))" />
          ))}
          <LabelList
            dataKey="p"
            position="right"
            formatter={(v: number | string) => `${Number(v).toFixed(1)}%`}
            style={{ fill: "rgb(var(--muted))", fontSize: 11 }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
