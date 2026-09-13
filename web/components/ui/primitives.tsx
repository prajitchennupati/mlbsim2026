import Link from "next/link";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export function Card({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-border bg-surface p-5 shadow-card",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CardTitle({
  children,
  right,
}: {
  children: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="mb-3 flex items-center justify-between gap-3">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">
        {children}
      </h2>
      {right ? <div className="text-xs text-muted">{right}</div> : null}
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="mb-7 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-[1.7rem] font-bold leading-tight tracking-tight">{title}</h1>
        {subtitle ? <p className="mt-1 text-sm text-muted">{subtitle}</p> : null}
      </div>
      {right}
    </div>
  );
}

export function StatTile({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "default" | "success" | "danger" | "accent";
}) {
  const tones = {
    default: "",
    success: "text-success",
    danger: "text-danger",
    accent: "text-accent",
  } as const;
  return (
    <div className="rounded-xl border border-border bg-surface px-4 py-3 shadow-card">
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
        {label}
      </div>
      <div className={cn("tabular mt-1 text-xl font-semibold", tones[tone])}>{value}</div>
      {hint ? <div className="mt-0.5 text-xs text-muted">{hint}</div> : null}
    </div>
  );
}

export function Badge({
  children,
  tone = "muted",
  className,
}: {
  children: ReactNode;
  tone?: "muted" | "home" | "away" | "accent" | "success" | "danger" | "warning";
  className?: string;
}) {
  const tones = {
    muted: "bg-border/60 text-fg",
    home: "bg-home/15 text-home",
    away: "bg-away/15 text-away",
    accent: "bg-accent/15 text-accent",
    success: "bg-success/15 text-success",
    danger: "bg-danger/15 text-danger",
    warning: "bg-warning/15 text-warning",
  } as const;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-dashed border-border bg-surface/50 p-10 text-center text-sm text-muted">
      {children}
    </div>
  );
}

export function LiveDot({ label = "Live" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-danger">
      <span className="h-2 w-2 rounded-full bg-danger animate-pulse-dot" />
      {label}
    </span>
  );
}

/**
 * A row of links styled as a segmented control. `current` is the href of the
 * active segment.
 */
export function Segmented({
  items,
  current,
}: {
  items: { href: string; label: string }[];
  current: string;
}) {
  return (
    <div className="inline-flex rounded-lg border border-border bg-surface p-0.5 text-sm">
      {items.map((it) => (
        <Link
          key={it.href}
          href={it.href}
          className={cn(
            "rounded-md px-3 py-1 font-medium transition",
            it.href === current
              ? "bg-accent text-white"
              : "text-muted hover:text-fg",
          )}
        >
          {it.label}
        </Link>
      ))}
    </div>
  );
}

/**
 * Two-colour probability bar with the favoured side's share labelled. `home` is
 * the home win probability in [0, 1].
 */
export function ProbBar({
  home,
  away,
  showTick = true,
}: {
  home: number;
  away: number;
  showTick?: boolean;
}) {
  const h = Math.round(home * 100);
  return (
    <div className="relative flex h-2.5 w-full overflow-hidden rounded-full bg-border">
      <div className="bg-away/80" style={{ width: `${100 - h}%` }} />
      <div className="bg-home/80" style={{ width: `${h}%` }} />
      {showTick ? (
        <span className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-surface/70" />
      ) : null}
    </div>
  );
}
