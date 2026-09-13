import { GamesBoard } from "@/components/GamesBoard";

// Home page — today's games board (see components/GamesBoard.tsx, shared
// with /games so the two can't drift apart).
export const revalidate = 120;

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<{ date?: string; team?: string; filter?: string }>;
}) {
  return await GamesBoard({ searchParams });
}
