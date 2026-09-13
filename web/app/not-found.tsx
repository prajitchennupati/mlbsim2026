import Link from "next/link";

export default function NotFound() {
  return (
    <div className="animate-fade-in-up py-24 text-center">
      <div className="mb-3 text-4xl animate-float">⚾</div>
      <h1 className="text-2xl font-bold">Not found</h1>
      <p className="mt-2 text-muted">That page, game, or team doesn&rsquo;t exist.</p>
      <Link
        href="/"
        className="mt-4 inline-block text-accent transition-transform duration-300 ease-premium hover:-translate-y-0.5 hover:underline"
      >
        Back home →
      </Link>
    </div>
  );
}
