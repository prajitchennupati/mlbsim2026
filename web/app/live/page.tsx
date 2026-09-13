import type { Metadata } from "next";
import { LiveBoard } from "@/components/LiveBoard";

// `/live` is an alias for the live board that also lives at `/`.
export const revalidate = 0;
// See app/page.tsx: gives a cold Render API room to wake up without Vercel's
// own function timeout racing ahead of it.
export const maxDuration = 60;

export const metadata: Metadata = { title: "Live board" };

export default async function LivePage() {
  return await LiveBoard();
}
