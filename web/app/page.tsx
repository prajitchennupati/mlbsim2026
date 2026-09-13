import { LiveBoard } from "@/components/LiveBoard";

// Live board — rendered per request; <AutoRefresh> re-pulls it on a timer.
export const revalidate = 0;

export default async function HomePage() {
  return await LiveBoard();
}
