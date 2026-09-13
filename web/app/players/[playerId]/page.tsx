import { PageHeader, Empty } from "@/components/ui/primitives";

export const revalidate = 600;

/**
 * Player pages need a dedicated `/players/{id}` API endpoint (rolling projections,
 * accuracy history, advanced metrics) — that lands with the resolver job. For now
 * this is a stable placeholder so links do not 404.
 */
export default async function PlayerPage({ params }: { params: Promise<{ playerId: string }> }) {
  const { playerId } = await params;
  return (
    <div className="space-y-6">
      <PageHeader title={`Player #${playerId}`} subtitle="Projections & accuracy history" />
      <Empty>
        Per-player pages arrive once the <code>/players/{"{id}"}</code> endpoint is wired
        (rolling projections + prediction accuracy). Per-game batter and pitcher projections are
        already on each game page.
      </Empty>
    </div>
  );
}
