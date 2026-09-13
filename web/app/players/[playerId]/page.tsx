import { notFound } from "next/navigation";
import Link from "next/link";
import { apiGetOrNull } from "@/lib/api";
import type { PlayerCard } from "@/lib/types";
import { num, pct, shortDate } from "@/lib/format";
import { Card, CardTitle, Empty, PageHeader, StatTile, stagger } from "@/components/ui/primitives";

export const revalidate = 300;

const STAT_LABELS: Record<string, string> = {
  proj_ip_per_start: "Proj IP / start",
  proj_so_per_start: "Proj K / start",
  proj_bb_per_start: "Proj BB / start",
  proj_hr_per_start: "Proj HR / start",
  proj_era_equiv: "Proj ERA (FIP-based)",
  p_6plus_k: "P(6+ K)",
  p_6plus_ip: "P(6+ IP)",
  n_seasons_used: "Seasons used",
};

export default async function PlayerPage({
  params,
}: {
  params: Promise<{ playerId: string }>;
}) {
  const { playerId } = await params;
  const card = await apiGetOrNull<PlayerCard>(`/players/${playerId}`);
  if (!card) notFound();

  const pitchingLine = card.season_lines.find((l) => l.group === "pitching");

  return (
    <div className="space-y-8">
      <PageHeader
        title={card.full_name ?? `Player #${card.player_id}`}
        subtitle={
          <span className="tabular">
            #{card.player_id}
            {card.primary_pos ? ` · ${card.primary_pos}` : ""}
            {card.throws ? ` · throws ${card.throws}` : ""}
          </span>
        }
      />

      {pitchingLine ? (
        <section>
          <CardTitle>
            Projection
            <span className="ml-2 font-mono text-fg">{pitchingLine.season}</span>
          </CardTitle>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {Object.entries(pitchingLine.stats).map(([key, value], i) => (
              <StatTile
                key={key}
                style={stagger(i)}
                label={STAT_LABELS[key] ?? key}
                value={key.startsWith("p_") ? pct(value, 1) : num(value, 2)}
              />
            ))}
          </div>
        </section>
      ) : (
        <Empty>No projection on file for this player yet.</Empty>
      )}

      {card.recent_predictions.length ? (
        <Card>
          <CardTitle>Recent starts</CardTitle>
          <div className="scroll-x">
            <table className="tabular w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="py-1.5">Date</th>
                  <th className="px-2 py-1.5">Game</th>
                  <th className="px-2 py-1.5 text-right">Proj IP</th>
                  <th className="px-2 py-1.5 text-right">Proj K</th>
                  <th className="px-2 py-1.5 text-right">P(6+K)</th>
                </tr>
              </thead>
              <tbody>
                {card.recent_predictions.map((p) => (
                  <tr
                    key={p.pred_id}
                    className="border-t border-border transition-colors duration-200 hover:bg-surface-2"
                  >
                    <td className="py-2 text-muted">{shortDate(p.game_date)}</td>
                    <td className="px-2 py-2">
                      <Link
                        href={`/games/${p.game_pk}`}
                        className="transition-colors duration-200 hover:text-accent"
                      >
                        #{p.game_pk}
                      </Link>
                    </td>
                    <td className="px-2 py-2 text-right">{num(p.proj.mean_ip, 1)}</td>
                    <td className="px-2 py-2 text-right">{num(p.proj.mean_k, 1)}</td>
                    <td className="px-2 py-2 text-right">{pct(p.prob.p_6plus_k)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}
    </div>
  );
}
