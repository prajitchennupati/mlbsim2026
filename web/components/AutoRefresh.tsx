"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

/**
 * Re-runs the server component tree on an interval (and when the tab regains
 * focus) so a page rendered with `revalidate = 0` stays live without a full
 * reload. Purely presentational otherwise — all data fetching stays on the server.
 */
export function AutoRefresh({
  intervalMs = 60_000,
  label = "Auto-updating",
}: {
  intervalMs?: number;
  label?: string;
}) {
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [last, setLast] = useState(() => Date.now());
  const [tick, setTick] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = () => startTransition(() => {
    router.refresh();
    setLast(Date.now());
  });

  useEffect(() => {
    timer.current = setInterval(refresh, intervalMs);
    const onVis = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVis);
    const clock = setInterval(() => setTick((t) => t + 1), 15_000);
    return () => {
      if (timer.current) clearInterval(timer.current);
      clearInterval(clock);
      document.removeEventListener("visibilitychange", onVis);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs]);

  const secs = Math.round((Date.now() - last) / 1000);
  const ago =
    secs < 20 ? "just now" : secs < 90 ? "a minute ago" : `${Math.round(secs / 60)}m ago`;

  return (
    <button
      onClick={refresh}
      className="inline-flex items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1 text-xs text-muted transition hover:text-fg"
      title="Refresh now"
      data-tick={tick}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          pending ? "bg-accent animate-pulse-dot" : "bg-success"
        }`}
      />
      {pending ? "Updating…" : `${label} · ${ago}`}
    </button>
  );
}
