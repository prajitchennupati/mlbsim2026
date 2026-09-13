import Link from "next/link";

export default function NotFound() {
  return (
    <div className="py-24 text-center">
      <h1 className="text-2xl font-bold">Not found</h1>
      <p className="mt-2 text-muted">That page, game, or team doesn&rsquo;t exist.</p>
      <Link href="/" className="mt-4 inline-block text-accent hover:underline">
        Back home →
      </Link>
    </div>
  );
}
