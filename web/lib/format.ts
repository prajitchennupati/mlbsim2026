export function pct(p: number | null | undefined, digits = 0): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

export function shortDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso.length <= 10 ? `${iso}T00:00:00` : iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function isoDate(d = new Date()): string {
  return d.toISOString().slice(0, 10);
}

/** "8:05 PM" in the viewer's locale, or "" when there is no start time. */
export function clockTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

/** Compact "2h ago" / "3d ago" from an ISO timestamp. */
export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const s = Math.round((Date.now() - then) / 1000);
  if (s < 60) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

/** Bucket a {"k": p} distribution into an array of {value, p} for charts. */
export function distToArray(
  dist: Record<string, number> | null | undefined,
  max = 12,
): { value: number; p: number }[] {
  if (!dist) return [];
  return Object.entries(dist)
    .map(([k, p]) => ({ value: Number(k), p }))
    .filter((d) => Number.isFinite(d.value) && d.value <= max)
    .sort((a, b) => a.value - b.value);
}
