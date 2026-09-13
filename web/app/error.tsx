"use client";

import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="animate-fade-in-up py-24 text-center">
      <h1 className="text-2xl font-bold">Something went wrong</h1>
      <p className="mt-2 text-sm text-muted">
        The API may be unavailable or still warming up.
      </p>
      <button
        onClick={reset}
        className="mt-4 rounded-md border border-border bg-surface px-3 py-1.5 text-sm shadow-card transition-all duration-300 ease-premium hover:-translate-y-0.5 hover:shadow-card-hover"
      >
        Try again
      </button>
    </div>
  );
}
