import Link from "next/link";
import { apiGetOrDefault, apiGetOrNull } from "@/lib/api";
import type { PredictionHistoryRow, PredictionSummary } from "@/lib/types";
import { num, pct, shortDate } from "@/lib/format";
import { Card, CardTitle, Empty, PageHeader, StatTile, Segmented } from "@/components/ui/primitives";
import { VerdictBadge } from "@/components/VerdictBadge";

export const revalidate = 120;

type SP = { model?: string; team?: string; view?: string };

export default async function PredictionsPage({
  searchParams,
}: {
  searchParams: Promise<SP>;
}) {
  const sp = await searchParams;
  const view = sp.view === "pending" ? "pending" : sp.view === "all" ? "all" : "graded";

  const histQs = new URLSearchParams({ limit: "400" });
  if (sp.model) histQs.set("model", sp.model);
  if (sp.team) histQs.set("team", sp.team);
  if (view === "graded") histQs.set("resolved", "true");
  if (view === "pending") histQs.set("resolved", "false");

  const [rows, summary] = await Promise.all([
    apiGetOrDefault<PredictionHistoryRow[]>(`/predictions/history?${histQs}`, []),
    apiGetOrNull<PredictionSummary>(
      `/predictions/summary${sp.model ? `?model=${sp.model}` : ""}`,
    ),
  ]);

  const base = sp.model ? `?model=${sp.model}&` : "?";

  return (
    <div className="space-y-7">
      <PageHeader
        title="Scorecard"
        subtitle="Every game prediction, graded against the final — no cherry-picking."
        right={
          <Segmented
            current={`${base}view=${view}`}
            items={[
              { href: `${base}view=graded`, label: "Graded" },
              { href: `${base}view=pending`, label: "Pending" },
              { href: `${base}view=all`, label: "All" },
            ]}
          />
        }
      />

      {summary ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile
            label="Accuracy"
            value={pct(summary.overall.accuracy, 1)}
            hint={`${summary.overall.correct}/${summary.overall.n}`}
            tone={
              summary.overall.accuracy != null && summary.overall.accuracy >= 0.5
                ? "success"
                : "default"
            }
          />
          <StatTile label="Last 7 days" value={pct(summary.last_7d.accuracy, 1)} hint={`${summary.last_7d.n} games`} />
          <StatTile label="Brier" value={num(summary.overall.brier, 3)} hint="vs 0.25 coin flip" />
          <StatTile
            label="Log loss"
            value={num(summary.overall.log_loss, 3)}
            hint={`vs ${summary.coin_flip_log_loss.toFixed(3)}`}
            tone={
              summary.overall.log_loss != null &&
              summary.overall.log_loss < summary.coin_flip_log_loss
                ? "success"
                : "danger"
            }
          />
        </div>
      ) : null}

      {summary?.by_model?.length ? (
        <Card>
          <CardTitle right={`${summary.pending} predictions awaiting a final`}>
            By model
          </CardTitle>
          <div className="scroll-x">
            <table className="tabular w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="py-1.5">Model</th>
                  <th className="px-2 py-1.5 text-right">Graded</th>
                  <th className="px-2 py-1.5 text-right">Accuracy</th>
                  <th className="px-2 py-1.5 text-right">Brier</th>
                  <th className="px-2 py-1.5 text-right">Log loss</th>
                </tr>
              </thead>
              <tbody>
                {summary.by_model.map((b) => (
                  <tr key={b.label} className="border-t border-border">
                    <td className="py-2 font-mono">
                      <Link href={`?model=${b.label}`} className="hover:text-accent">
                        {b.label}
                      </Link>
                    </td>
                    <td className="px-2 py-2 text-right">{b.n}</td>
                    <td className="px-2 py-2 text-right font-semibold">
                      {pct(b.accuracy, 1)}
                    </td>
                    <td className="px-2 py-2 text-right text-muted">{num(b.brier, 3)}</td>
                    <td
                      className={`px-2 py-2 text-right ${
                        b.log_loss != null && b.log_loss < summary.coin_flip_log_loss
                          ? "text-success"
                          : "text-muted"
                      }`}
                    >
                      {num(b.log_loss, 3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}

      <div>
        <CardTitle>
          {view === "graded"
            ? "Graded predictions"
            : view === "pending"
              ? "Awaiting a final"
              : "All predictions"}
          <span className="ml-2 font-mono text-fg">{rows.length}</span>
        </CardTitle>
        {rows.length ? (
          <div className="scroll-x rounded-2xl border border-border shadow-card">
            <table className="tabular w-full text-sm">
              <thead className="bg-surface-2 text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="px-3 py-2.5">Date</th>
                  <th className="px-3 py-2.5">Matchup</th>
                  <th className="px-3 py-2.5">Pick</th>
                  <th className="px-3 py-2.5 text-right">Conf.</th>
                  <th className="px-3 py-2.5 text-right">Proj</th>
                  <th className="px-3 py-2.5 text-right">Final</th>
                  <th className="px-3 py-2.5">Result</th>
                  <th className="px-3 py-2.5 text-right">Brier</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const away = r.away_abbr ?? "AWY";
                  const home = r.home_abbr ?? "HOM";
                  const pick = r.predicted_winner === "home" ? home : away;
                  const conf =
                    r.predicted_winner === "home"
                      ? r.home_win_prob
                      : 1 - r.home_win_prob;
                  return (
                    <tr key={r.pred_id} className="border-t border-border hover:bg-surface-2">
                      <td className="px-3 py-2 text-muted">{shortDate(r.game_date)}</td>
                      <td className="px-3 py-2">
                        <Link href={`/games/${r.game_pk}`} className="hover:text-accent">
                          {away} @ {home}
                        </Link>
                      </td>
                      <td className="px-3 py-2 font-semibold">{pick}</td>
                      <td className="px-3 py-2 text-right text-muted">{pct(conf, 0)}</td>
                      <td className="px-3 py-2 text-right text-muted">
                        {r.exp_away_runs != null
                          ? `${num(r.exp_away_runs)}–${num(r.exp_home_runs)}`
                          : "—"}
                      </td>
                      <td className="px-3 py-2 text-right">
                        {r.home_score != null
                          ? `${r.away_score}–${r.home_score}`
                          : "—"}
                      </td>
                      <td className="px-3 py-2">
                        <VerdictBadge
                          v={{
                            is_final: r.home_score != null,
                            correct: r.correct,
                            predicted_winner: r.predicted_winner,
                            home_win_prob: r.home_win_prob,
                          }}
                          homeAbbr={home}
                          awayAbbr={away}
                        />
                      </td>
                      <td className="px-3 py-2 text-right text-muted">{num(r.brier, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>Nothing here yet for this filter.</Empty>
        )}
      </div>
    </div>
  );
}
