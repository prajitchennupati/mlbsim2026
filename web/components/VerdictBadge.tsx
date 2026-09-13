import { Badge } from "@/components/ui/primitives";
import { cn } from "@/lib/cn";

type Verdict = {
  is_final: boolean;
  correct?: boolean | null;
  predicted_winner?: "home" | "away" | null;
  home_win_prob?: number | null;
};

/**
 * The grade for a single game prediction:
 *   • not final yet        -> "PICK: HOME 61%"
 *   • final, not resolved  -> "AWAITING GRADE"
 *   • final + resolved     -> "CORRECT" / "MISS"
 */
export function VerdictBadge({
  v,
  homeAbbr,
  awayAbbr,
  size = "sm",
}: {
  v: Verdict;
  homeAbbr: string;
  awayAbbr: string;
  size?: "sm" | "lg";
}) {
  const cls = size === "lg" ? "px-3 py-1 text-sm" : "";

  if (v.is_final && (v.correct === true || v.correct === false)) {
    return v.correct ? (
      <Badge tone="success" className={cls}>
        ✓ Correct
      </Badge>
    ) : (
      <Badge tone="danger" className={cls}>
        ✗ Miss
      </Badge>
    );
  }

  if (v.is_final) {
    return (
      <Badge tone="warning" className={cls}>
        Awaiting grade
      </Badge>
    );
  }

  if (v.predicted_winner && v.home_win_prob != null) {
    const p = v.predicted_winner === "home" ? v.home_win_prob : 1 - v.home_win_prob;
    const abbr = v.predicted_winner === "home" ? homeAbbr : awayAbbr;
    return (
      <Badge
        tone={v.predicted_winner === "home" ? "home" : "away"}
        className={cn(cls, "font-mono")}
      >
        Pick {abbr} {Math.round(p * 100)}%
      </Badge>
    );
  }

  return (
    <Badge tone="muted" className={cls}>
      No pick
    </Badge>
  );
}
