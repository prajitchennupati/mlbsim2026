import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { EasterEgg } from "@/components/EasterEgg";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: process.env.NEXT_PUBLIC_SITE_URL
    ? new URL(process.env.NEXT_PUBLIC_SITE_URL)
    : undefined,
  title: {
    default: "Diamond Signal — live AI MLB predictions & World Series odds",
    template: "%s · Diamond Signal",
  },
  description:
    "Every MLB game predicted — win probability, projected score, and player stat lines from a plate-appearance Monte Carlo — then graded correct or wrong once the game goes final.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-screen font-sans antialiased">
        <Nav />
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
        <footer className="mx-auto max-w-6xl px-4 py-10 text-xs leading-relaxed text-muted">
          Non-commercial research project. Data: MLB Stats API, Baseball Savant, Retrosheet,
          FanGraphs, Chadwick Bureau. Predictions are model output, graded honestly against
          results — not betting advice.
        </footer>
        <EasterEgg />
      </body>
    </html>
  );
}
