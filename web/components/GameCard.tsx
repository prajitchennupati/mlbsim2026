import Link from "next/link";
import type { GameSummary } from "@/lib/types";
import { clockTime, num, pct } from "@/lib/format";
import { ProbBar } from "@/components/ui/primitives";
import { VerdictBadge } from "@/components/VerdictBadge";
import { cn } from "@/lib/cn";

export function GameCard({ g }: { g: GameSummary }) {
  const home = g.home_abbr ?? String(g.home_team_id);
  const away = g.away_abbr ?? String(g.away_team_id);
  const hp = g.home_win_prob ?? null;
  const final = g.is_final;
  const homeWon = final && (g.home_score ?? 0) > (g.away_score ?? 0);
  const missed = final && g.correct === false;

  return (
    <Link
      href={`/games/${g.game_pk}`}
      className={cn(
        "group block animate-fade-in rounded-2xl border bg-surface p-4 shadow-card transition hover:-translate-y-0.5 hover:border-accent/50",
        missed ? "border-danger/30" : "border-border",
      )}
    >
      <div className="flex items-center justify-between gap-2 text-xs text-muted">
        <span>
          {final ? "Final" : g.status || clockTime(g.start_time_utc) || "Scheduled"}
        </span>
        <VerdictBadge v={g} homeAbbr={home} awayAbbr={away} />
      </div>

      <div className="tabular mt-3 space-y-1.5">
        <TeamRow
          abbr={away}
          side="away"
          score={g.away_score}
          prob={hp != null ? 1 - hp : null}
          isWinner={final && !homeWon}
          isPick={g.predicted_winner === "away"}
          final={final}
        />
        <TeamRow
          abbr={home}
          side="home"
          score={g.home_score}
          prob={hp}
          isWinner={homeWon}
          isPick={g.predicted_winner === "home"}
          final={final}
        />
      </div>

      {hp != null ? (
        <div className={cn("mt-3", final && "opacity-40")}>
          <ProbBar home={hp} away={1 - hp} />
        </div>
      ) : null}

      <div className="mt-2.5 flex items-center justify-between text-xs text-muted">
        {g.exp_home_runs != null ? (
          <span>
            proj {num(g.exp_away_runs)}–{num(g.exp_home_runs)}
          </span>
        ) : (
          <span />
        )}
        {g.model_id ? <span className="font-mono">{g.model_id}</span> : null}
      </div>
    </Link>
  );
}

function TeamRow({
  abbr,
  side,
  score,
  prob,
  isWinner,
  isPick,
  final,
}: {
  abbr: string;
  side: "home" | "away";
  score?: number | null;
  prob: number | null;
  isWinner: boolean;
  isPick: boolean;
  final: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="flex items-center gap-2">
        <span
          className={cn(
            "h-2.5 w-2.5 rounded-full",
            side === "home" ? "bg-home" : "bg-away",
          )}
        />
        <span
          className={cn(
            "font-semibold",
            final && !isWinner && "text-muted",
            side === "home" ? "text-home" : "text-away",
          )}
        >
          {abbr}
        </span>
        {isPick ? (
          <span className="rounded bg-accent/15 px-1 text-[10px] font-bold uppercase tracking-wide text-accent">
            pick
          </span>
        ) : null}
      </span>
      <span className="flex items-center gap-3 text-sm">
        {prob != null ? (
          <span className={cn("text-muted", final && "opacity-60")}>{pct(prob)}</span>
        ) : null}
        {score != null ? (
          <span
            className={cn(
              "w-5 text-right font-bold",
              isWinner ? "text-fg" : "text-muted",
            )}
          >
            {score}
          </span>
        ) : null}
      </span>
    </div>
  );
}
