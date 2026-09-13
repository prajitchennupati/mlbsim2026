"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

type Bin = { mean_pred: number; mean_actual: number; n: number };

export function CalibrationPlot({ bins }: { bins: Bin[] }) {
  if (!bins?.length) {
    return <div className="py-6 text-center text-sm text-muted">no calibration data</div>;
  }
  const data = [...bins]
    .sort((a, b) => a.mean_pred - b.mean_pred)
    .map((b) => ({ x: b.mean_pred, y: b.mean_actual, n: b.n }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
        <CartesianGrid stroke="rgb(var(--border))" strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          domain={[0, 1]}
          tick={{ fontSize: 11, fill: "rgb(var(--muted))" }}
          label={{ value: "predicted", position: "insideBottom", offset: -4, fontSize: 10 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={[0, 1]}
          tick={{ fontSize: 11, fill: "rgb(var(--muted))" }}
          label={{ value: "actual", angle: -90, position: "insideLeft", fontSize: 10 }}
        />
        <ZAxis type="number" dataKey="n" range={[40, 260]} />
        <ReferenceLine
          segment={[
            { x: 0, y: 0 },
            { x: 1, y: 1 },
          ]}
          stroke="rgb(var(--muted))"
          strokeDasharray="4 4"
        />
        <Tooltip
          formatter={(v, k) => [Number(v).toFixed(3), k === "y" ? "actual" : "predicted"]}
          contentStyle={{
            background: "rgb(var(--surface))",
            border: "1px solid rgb(var(--border))",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Scatter data={data} fill="rgb(var(--accent))" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function LineSeries({
  data,
  xKey,
  yKey,
}: {
  data: Record<string, number | string>[];
  xKey: string;
  yKey: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -16 }}>
        <CartesianGrid stroke="rgb(var(--border))" strokeDasharray="3 3" />
        <XAxis dataKey={xKey} tick={{ fontSize: 11, fill: "rgb(var(--muted))" }} />
        <YAxis tick={{ fontSize: 11, fill: "rgb(var(--muted))" }} />
        <Tooltip
          contentStyle={{
            background: "rgb(var(--surface))",
            border: "1px solid rgb(var(--border))",
            borderRadius: 8,
            fontSize: 12,
          }}
        />
        <Line
          type="monotone"
          dataKey={yKey}
          stroke="rgb(var(--accent))"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
