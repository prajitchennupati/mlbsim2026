import type { Metadata } from "next";
import { LiveBoard } from "@/components/LiveBoard";

// `/live` is an alias for the live board that also lives at `/`.
export const revalidate = 0;

export const metadata: Metadata = { title: "Live board" };

export default async function LivePage() {
  return await LiveBoard();
}
