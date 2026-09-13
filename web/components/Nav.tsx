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
    <header className="sticky top-0 z-20 border-b border-border bg-bg/80 backdrop-blur">
      <nav className="mx-auto flex max-w-6xl items-center gap-1 px-4 py-3">
        <Link href="/" className="mr-3 shrink-0 font-bold tracking-tight">
          <span className="text-accent">MLB</span>Playoffs
          <span className="text-muted">2026</span>
        </Link>
        <div className="scroll-x flex gap-1 text-sm">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                "shrink-0 rounded-md px-2.5 py-1 font-medium transition",
                isActive(l.href)
                  ? "bg-surface text-fg shadow-card"
                  : "text-muted hover:bg-surface hover:text-fg",
              )}
            >
              {l.label === "Live" ? (
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full bg-danger animate-pulse-dot" />
                  {l.label}
                </span>
              ) : (
                l.label
              )}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
}
