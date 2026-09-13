import type { Factor } from "@/lib/types";

export function FactorList({
  factors,
  homeAbbr = "home",
  awayAbbr = "away",
}: {
  factors: Factor[];
  homeAbbr?: string;
  awayAbbr?: string;
}) {
  if (!factors?.length) {
    return <p className="text-sm text-muted">No factor breakdown available.</p>;
  }
  return (
    <ul className="space-y-2">
      {factors.map((f) => {
        const favTeam = f.favours === "home" ? homeAbbr : awayAbbr;
        const favClass = f.favours === "home" ? "text-home" : "text-away";
        return (
          <li key={f.feature} className="flex items-center justify-between gap-3 text-sm">
            <span>{f.label}</span>
            <span className="tabular flex items-center gap-2 whitespace-nowrap">
              <span className={favClass}>{favTeam}</span>
              <span className="text-muted">
                {f.prob_shift >= 0 ? "+" : ""}
                {(f.prob_shift * 100).toFixed(1)}%
              </span>
            </span>
          </li>
        );
      })}
    </ul>
  );
}
