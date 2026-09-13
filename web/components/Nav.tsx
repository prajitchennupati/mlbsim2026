"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useRef, useState } from "react";
import { cn } from "@/lib/cn";

const LINKS = [
  { href: "/games", label: "Games" },
  { href: "/predictions", label: "Scorecard" },
  { href: "/teams", label: "Teams" },
  { href: "/standings", label: "Standings" },
  { href: "/playoffs", label: "Playoffs" },
  { href: "/model", label: "Model" },
] as const;

const STRETCH_CLICKS = 7;
const STRETCH_WINDOW_MS = 1500;

export function Nav() {
  const pathname = usePathname() ?? "/";
  // "/" is now the games board's home, so the Games tab covers both.
  const isActive = (href: string) =>
    href === "/games" ? pathname === "/" || pathname.startsWith("/games") : pathname.startsWith(href);

  // Easter egg: click the logo 7x fast for the "7th inning stretch."
  const clickTimes = useRef<number[]>([]);
  const [stretching, setStretching] = useState(false);
  const handleLogoClick = () => {
    const now = Date.now();
    clickTimes.current = [...clickTimes.current, now].filter((t) => now - t < STRETCH_WINDOW_MS);
    if (clickTimes.current.length >= STRETCH_CLICKS) {
      clickTimes.current = [];
      setStretching(true);
      window.setTimeout(() => setStretching(false), 2200);
    }
  };

  return (
    <header className="sticky top-0 z-20 border-b border-border/80 bg-bg/75 backdrop-blur-lg backdrop-saturate-150">
      <nav className="mx-auto flex max-w-6xl items-center gap-1 px-4 py-3">
        <Link
          href="/"
          onClick={handleLogoClick}
          className="group relative mr-4 flex shrink-0 items-center gap-2 font-bold tracking-tight transition-transform duration-300 ease-premium hover:-translate-y-px"
        >
          <span
            className="relative flex h-7 w-7 items-center justify-center overflow-hidden rounded-lg bg-gradient-to-br from-accent to-accent/60 text-[13px] font-black text-white shadow-glow transition-transform duration-500 ease-premium group-hover:rotate-[8deg]"
            style={stretching ? { animation: "egg-logo-pop 1.1s ease-in-out 2" } : undefined}
          >
            ⚾
          </span>
          <span>
            Diamond<span className="text-accent">Signal</span>
          </span>
          {stretching ? (
            <span className="pointer-events-none absolute left-0 top-full mt-2 animate-fade-in-up whitespace-nowrap rounded-full border border-border bg-surface px-3 py-1 text-xs font-semibold normal-case tracking-normal text-fg shadow-pop">
              🎶 Take Me Out to the Ball Game! (7th inning stretch)
            </span>
          ) : null}
        </Link>
        <div className="scroll-x flex gap-1 text-sm">
          {LINKS.map((l) => {
            const active = isActive(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={cn(
                  "group relative shrink-0 rounded-md px-2.5 py-1.5 font-medium transition-all duration-300 ease-premium",
                  active ? "text-fg" : "text-muted hover:text-fg",
                )}
              >
                {active ? (
                  <span className="absolute inset-0 -z-10 animate-scale-in rounded-md bg-surface shadow-card" />
                ) : (
                  <span className="absolute inset-0 -z-10 rounded-md bg-surface-2 opacity-0 transition-opacity duration-300 ease-premium group-hover:opacity-100" />
                )}
                {l.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
