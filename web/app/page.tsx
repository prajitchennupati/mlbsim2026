import { LiveBoard } from "@/components/LiveBoard";

// Live board — rendered per request; <AutoRefresh> re-pulls it on a timer.
export const revalidate = 0;
// Give the free-tier Render API room to wake from a cold sleep (up to ~50s)
// without Vercel's own function timeout killing the request first — that
// race, not a real crash, was the previous "Something went wrong" cause.
export const maxDuration = 60;

export default async function HomePage() {
  return await LiveBoard();
}
