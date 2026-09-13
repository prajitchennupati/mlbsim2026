import Link from "next/link";
import { apiGet } from "@/lib/api";
import type { GameSummary } from "@/lib/types";
import { isoDate } from "@/lib/format";
import { Empty, PageHeader, Segmented } from "@/components/ui/primitives";
import { GameCard } from "@/components/GameCard";

export const revalidate = 120;

export default async function GamesPage({
  searchParams,
}: {
  searchParams: Promise<{ date?: string; team?: string; filter?: string }>;
}) {
  const sp = await searchParams;
  const date = sp.date ?? isoDate();
  const filter = sp.filter === "final" ? "final" : sp.filter === "upcoming" ? "upcoming" : "all";
  const qs = new URLSearchParams({ date });
  if (sp.team) qs.set("team", sp.team);

  const all = await apiGet<GameSummary[]>(`/games?${qs}`).catch(() => [] as GameSummary[]);
  const games =
    filter === "final"
      ? all.filter((g) => g.is_final)
      : filter === "upcoming"
        ? all.filter((g) => !g.is_final)
        : all;

  const graded = all.filter((g) => g.correct != null);
  const right = graded.filter((g) => g.correct).length;
  const prev = shift(date, -1);
  const next = shift(date, 1);
  const teamQ = sp.team ? `&team=${sp.team}` : "";

  return (
    <div className="space-y-6">
      <PageHeader
        title="Games"
        subtitle={
          <>
            {sp.team ? `${sp.team} · ` : ""}
            {date}
            {graded.length ? (
              <span className="ml-2 font-mono">
                · {right}/{graded.length} picks correct
              </span>
            ) : null}
          </>
        }
        right={
          <div className="tabular flex items-center gap-2 text-sm">
            <Link
              href={`/games?date=${prev}${teamQ}`}
              className="rounded-md border border-border bg-surface px-2 py-1 hover:border-accent/50"
            >
              ← {prev.slice(5)}
            </Link>
            <Link
              href={`/games?date=${next}${teamQ}`}
              className="rounded-md border border-border bg-surface px-2 py-1 hover:border-accent/50"
            >
              {next.slice(5)} →
            </Link>
          </div>
        }
      />

      <Segmented
        current={`/games?date=${date}${teamQ}&filter=${filter}`}
        items={[
          { href: `/games?date=${date}${teamQ}&filter=all`, label: `All ${all.length}` },
          {
            href: `/games?date=${date}${teamQ}&filter=upcoming`,
            label: "Scheduled / live",
          },
          { href: `/games?date=${date}${teamQ}&filter=final`, label: "Final" },
        ]}
      />

      {games.length ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {games.map((g) => (
            <GameCard key={g.game_pk} g={g} />
          ))}
        </div>
      ) : (
        <Empty>No games match this filter for {date}.</Empty>
      )}
    </div>
  );
}

function shift(iso: string, days: number) {
  const d = new Date(`${iso}T00:00:00`);
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}
