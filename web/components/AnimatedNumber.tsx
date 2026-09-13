"use client";

import { useEffect, useRef, useState } from "react";

const EASE_OUT_EXPO = (t: number) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t));

/**
 * Counts up from 0 to `value` on mount (and re-animates whenever `value`
 * changes, e.g. an auto-refresh landing a new accuracy number). Renders
 * through `format` so callers keep control of "%"/decimals/units.
 */
export function AnimatedNumber({
  value,
  format,
  durationMs = 900,
}: {
  value: number | null | undefined;
  format: (n: number) => string;
  durationMs?: number;
}) {
  const [display, setDisplay] = useState(0);
  const from = useRef(0);
  const raf = useRef<number | null>(null);

  useEffect(() => {
    if (value === null || value === undefined || Number.isNaN(value)) return;
    const start = performance.now();
    const startValue = from.current;
    const target = value;

    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = EASE_OUT_EXPO(t);
      setDisplay(startValue + (target - startValue) * eased);
      if (t < 1) {
        raf.current = requestAnimationFrame(tick);
      } else {
        from.current = target;
      }
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, durationMs]);

  if (value === null || value === undefined || Number.isNaN(value)) {
    return <span className="tabular">—</span>;
  }
  return <span className="tabular">{format(display)}</span>;
}
