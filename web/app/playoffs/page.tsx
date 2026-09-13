import { apiGetOrNull } from "@/lib/api";
import type { PlayoffsOut } from "@/lib/types";
import { Card, CardTitle, Empty, PageHeader, stagger } from "@/components/ui/primitives";
import { WSOddsBar } from "@/components/charts/WSOddsBar";
import { Bracket } from "@/components/charts/Bracket";

export const revalidate = 600;

export default async function PlayoffsPage() {
  const data = await apiGetOrNull<PlayoffsOut>("/playoffs");

  return (
    <div className="space-y-6">
      <PageHeader
        title="Playoffs"
        subtitle={
          data
            ? `From a ${data.as_of ? `${data.as_of} ` : ""}season Monte Carlo (run ${data.sim_run_id})`
            : "Run a season simulation to populate this page"
        }
      />
      {!data ? (
        <Empty>No season simulation has been run yet.</Empty>
      ) : (
        <>
          <Card style={stagger(0)}>
            <CardTitle>World Series probability</CardTitle>
            <WSOddsBar teams={data.teams} limit={14} />
          </Card>
          <Card style={stagger(1)}>
            <CardTitle>Projected bracket</CardTitle>
            <Bracket data={data} />
          </Card>
        </>
      )}
    </div>
  );
}
