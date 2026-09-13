"use client";

import { useEffect, useState } from "react";

const KONAMI = [
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
const IDLE_MS = 3 * 60 * 1000;

type Celebration = { emoji: string; message: string; confetti: number; trot?: boolean };

// Typed anywhere on the page (no input focus needed) — checked as a rolling
// suffix of the last few keys pressed.
const PHRASES: Record<string, Celebration> = {
  walkoff: { emoji: "🚶💥", message: "WALK-OFF WIN!", confetti: 40 },
  playball: { emoji: "⚾", message: "Play ball!", confetti: 16 },
  moneyball: {
    emoji: "📊",
    message: "Sometimes the boring model wins — Elo's still undefeated against the fancy stuff.",
    confetti: 0,
  },
};

/**
 * Six easter eggs, all pure CSS/JS (no new dependency), all collapsing under
 * prefers-reduced-motion:
 *   1. Konami code (↑↑↓↓←→←→BA)         -> home-run trot + confetti
 *   2-4. type "walkoff" / "playball" / "moneyball" anywhere -> themed banner
 *   5. idle 3 minutes with no input       -> a ball bounces in the corner
 *   6. click the nav logo 7x fast         -> handled locally in Nav.tsx
 */
export function EasterEgg() {
  const [active, setActive] = useState<Celebration | null>(null);
  const [idleBall, setIdleBall] = useState(false);

  useEffect(() => {
    console.log(
      '%c⚾ psst — try the Konami code (↑↑↓↓←→←→BA), or type "walkoff", "playball", or "moneyball"',
      "font-size:12px;color:#8886f8",
    );
    let arrowPos = 0;
    let buffer = "";
    function onKey(e: KeyboardEvent) {
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;

      if (key === KONAMI[arrowPos]) {
        arrowPos++;
        if (arrowPos === KONAMI.length) {
          arrowPos = 0;
          fire({ emoji: "🎉", message: "You found the easter egg!", confetti: 28, trot: true });
        }
      } else {
        arrowPos = key === KONAMI[0] ? 1 : 0;
      }

      if (/^[a-z]$/.test(key)) {
        buffer = (buffer + key).slice(-12);
        for (const [phrase, celebration] of Object.entries(PHRASES)) {
          if (buffer.endsWith(phrase)) {
            buffer = "";
            fire(celebration);
            break;
          }
        }
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    let idleTimer: ReturnType<typeof setTimeout>;
    const wake = () => {
      setIdleBall(false);
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => setIdleBall(true), IDLE_MS);
    };
    wake();
    const events: (keyof WindowEventMap)[] = ["mousemove", "keydown", "scroll", "touchstart"];
    events.forEach((ev) => window.addEventListener(ev, wake, { passive: true }));
    return () => {
      clearTimeout(idleTimer);
      events.forEach((ev) => window.removeEventListener(ev, wake));
    };
  }, []);

  function fire(c: Celebration) {
    setActive(c);
    window.setTimeout(() => setActive(null), c.trot ? 2600 : 3200);
  }

  return (
    <>
      {active ? (
        <div className="pointer-events-none fixed inset-0 z-[100] overflow-hidden">
          {active.trot ? (
            <div
              className="absolute top-1/3 text-5xl"
              style={{ animation: "egg-trot 2.4s ease-in-out forwards" }}
            >
              ⚾
            </div>
          ) : null}
          {Array.from({ length: active.confetti }, (_, i) => (
            <span
              key={i}
              className={`absolute top-0 h-2 w-2 rounded-sm ${COLORS[i % COLORS.length]}`}
              style={{
                left: `${(i * 37) % 100}%`,
                animation: `egg-confetti-fall ${1.8 + (i % 5) * 0.2}s ease-in ${(i % 7) * 0.08}s forwards`,
              }}
            />
          ))}
          <div className="absolute bottom-10 left-1/2 max-w-[min(90vw,26rem)] -translate-x-1/2 animate-fade-in-up rounded-full border border-border bg-surface px-4 py-2 text-center text-sm font-semibold shadow-pop">
            {active.emoji} {active.message}
          </div>
        </div>
      ) : null}

      {idleBall ? (
        <div
          aria-hidden
          className="pointer-events-none fixed bottom-6 right-6 z-[90] text-3xl opacity-70 animate-float"
        >
          ⚾
        </div>
      ) : null}
    </>
  );
}
