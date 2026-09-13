export default function Loading() {
  return (
    <div className="space-y-6">
      <div className="skeleton h-9 w-56 rounded-lg" />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="skeleton h-20 rounded-xl border border-border" />
        ))}
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="skeleton h-40 rounded-2xl border border-border" />
        ))}
      </div>
    </div>
  );
}
