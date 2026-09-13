"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/cn";

const LINKS = [
  { href: "/live", label: "Live" },
  { href: "/games", label: "Games" },
  { href: "/predictions", label: "Scorecard" },
  { href: "/teams", label: "Teams" },
  { href: "/standings", label: "Standings" },
  { href: "/playoffs", label: "Playoffs" },
  { href: "/model", label: "Model" },
] as const;

export function Nav() {
  const pathname = usePathname() ?? "/";
  const isActive = (href: string) =>
    href === "/live" ? pathname === "/" || pathname.startsWith("/live") : pathname.startsWith(href);

  return (
    <header className="sticky top-0 z-20 border-b border-border/80 bg-bg/75 backdrop-blur-lg backdrop-saturate-150">
      <nav className="mx-auto flex max-w-6xl items-center gap-1 px-4 py-3">
        <Link
          href="/"
          className="group mr-4 flex shrink-0 items-center gap-2 font-bold tracking-tight transition-transform duration-300 ease-premium hover:-translate-y-px"
        >
          <span className="relative flex h-7 w-7 items-center justify-center overflow-hidden rounded-lg bg-gradient-to-br from-accent to-accent/60 text-[13px] font-black text-white shadow-glow transition-transform duration-500 ease-premium group-hover:rotate-[8deg]">
            ⚾
          </span>
          <span>
            Diamond<span className="text-accent">Signal</span>
          </span>
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
                {l.label === "Live" ? (
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-danger animate-pulse-dot" />
                    {l.label}
                  </span>
                ) : (
                  l.label
                )}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
