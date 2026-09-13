import { apiGetOrDefault, apiGetOrNull } from "@/lib/api";
import type { EvalOut, ModelOut } from "@/lib/types";
import { Card, CardTitle, PageHeader, stagger } from "@/components/ui/primitives";
import { CalibrationPlot } from "@/components/charts/CalibrationPlot";

export const revalidate = 900;

export default async function ModelPage() {
  const models = await apiGetOrDefault<ModelOut[]>("/models", []);
  const winModel =
    models.find((m) => m.model_id === "ensemble_v1") ??
    models.find((m) => m.model_id === "direct_v1") ??
    models[0];
  const evalOut = winModel
    ? await apiGetOrNull<EvalOut>(`/models/${winModel.model_id}/evaluation`)
    : null;

  return (
    <div className="space-y-8">
      <PageHeader title="How the model works" subtitle="Data, methods, validation, and honest limitations" />

      <Card>
        <CardTitle>The short version</CardTitle>
        <p className="text-sm leading-relaxed text-muted">
          One <strong className="text-fg">plate-appearance Monte Carlo simulator</strong> is the
          core: for each matchup it draws pitcher → batter events (a log5 combination of batter,
          pitcher and league rates), advances runners, and plays 10,000+ full games — so win
          probability, score distributions, runs by inning and player stat lines are all read off
          the same simulation and stay mutually consistent. A fast{" "}
          <strong className="text-fg">direct path</strong> (Elo + logistic + Poisson) feeds the
          100,000-season Monte Carlo, and a{" "}
          <strong className="text-fg">stacked, isotonic-calibrated ensemble</strong> blends the
          two.
        </p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card style={stagger(0)}>
          <CardTitle>Registered models</CardTitle>
          <table className="tabular w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase text-muted">
                <th className="py-1">Model</th>
                <th className="px-2 py-1">Kind</th>
                <th className="px-2 py-1">Trained through</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.model_id}
                  className="border-t border-border transition-colors duration-200 hover:bg-surface-2"
                >
                  <td className="py-1.5">
                    <span className="font-medium">{m.name}</span>
                    <span className="ml-2 font-mono text-xs text-muted">{m.model_id}</span>
                  </td>
                  <td className="px-2 py-1.5 text-muted">{m.kind}</td>
                  <td className="px-2 py-1.5 text-muted">{m.train_end ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card style={stagger(1)}>
          <CardTitle>Calibration {winModel ? `· ${winModel.model_id}` : ""}</CardTitle>
          <CalibrationPlot bins={evalOut?.calibration ?? []} />
          <p className="mt-2 text-xs text-muted">
            Points on the diagonal mean the stated probability matches reality. Size = games in
            the bin.
          </p>
        </Card>
      </div>

      <Card>
        <CardTitle>Validation &amp; leakage</CardTitle>
        <ul className="list-inside list-disc space-y-1 text-sm text-muted">
          <li>Time-ordered splits only — train ≤ 2022, validate 2023, test 2024, then walk-forward through 2025.</li>
          <li>Every feature is computed through a strict point-in-time join; a feature &ldquo;as of&rdquo; a date never changes when future rows arrive.</li>
          <li>Each model has a named baseline it must beat: Elo for win/loss, Marcel for player props.</li>
          <li>The LLM prediction experiment is reported either way — &ldquo;the LLM does not beat the GBM&rdquo; is an acceptable result.</li>
        </ul>
      </Card>

      {evalOut?.runs?.length ? (
        <Card>
          <CardTitle>Evaluation runs</CardTitle>
          <pre className="overflow-x-auto rounded-lg bg-bg p-3 text-xs text-muted">
            {JSON.stringify(evalOut.runs.slice(0, 5), null, 2)}
          </pre>
        </Card>
      ) : null}
    </div>
  );
}
