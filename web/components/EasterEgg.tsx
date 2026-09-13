"use client";

import { useEffect, useState } from "react";

const CODE = [
  "ArrowUp",
  "ArrowUp",
  "ArrowDown",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "ArrowLeft",
  "ArrowRight",
  "b",
  "a",
];
const COLORS = ["bg-accent", "bg-home", "bg-away", "bg-warning", "bg-success", "bg-danger"];
const CONFETTI = Array.from({ length: 28 }, (_, i) => i);

/**
 * Konami code (↑↑↓↓←→←→BA) triggers a home-run trot + confetti. Pure CSS
 * animation (keyframes live in globals.css) so there's no new dependency, and
 * it fully collapses under prefers-reduced-motion like everything else.
 */
export function EasterEgg() {
  const [active, setActive] = useState(false);

  useEffect(() => {
    console.log(
      "%c⚾ psst — try the Konami code (↑↑↓↓←→←→BA)",
      "font-size:12px;color:#8886f8",
    );
    let pos = 0;
    function onKey(e: KeyboardEvent) {
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      const expected = CODE[pos];
      if (key === expected) {
        pos++;
        if (pos === CODE.length) {
          pos = 0;
          setActive(true);
          window.setTimeout(() => setActive(false), 2600);
        }
      } else {
        pos = key === CODE[0] ? 1 : 0;
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!active) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-[100] overflow-hidden">
      <div
        className="absolute top-1/3 text-5xl"
        style={{ animation: "egg-trot 2.4s ease-in-out forwards" }}
      >
        ⚾
      </div>
      {CONFETTI.map((i) => (
        <span
          key={i}
          className={`absolute top-0 h-2 w-2 rounded-sm ${COLORS[i % COLORS.length]}`}
          style={{
            left: `${(i * 37) % 100}%`,
            animation: `egg-confetti-fall ${1.8 + (i % 5) * 0.2}s ease-in ${(i % 7) * 0.08}s forwards`,
          }}
        />
      ))}
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2 animate-fade-in-up whitespace-nowrap rounded-full border border-border bg-surface px-4 py-2 text-sm font-semibold shadow-pop">
        🎉 Walk-off! You found the easter egg.
      </div>
    </div>
  );
}
