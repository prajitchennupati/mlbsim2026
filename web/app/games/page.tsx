import { GamesBoard } from "@/components/GamesBoard";

export const revalidate = 120;

export default async function GamesPage({
  searchParams,
}: {
  searchParams: Promise<{ date?: string; team?: string; filter?: string }>;
}) {
  return await GamesBoard({ searchParams });
}
