const SOURCES = ["MLB Stats API", "Baseball Savant", "Retrosheet", "FanGraphs", "Chadwick Bureau"];

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="mt-16 border-t border-border">
      <div className="mx-auto max-w-6xl animate-fade-in px-4 py-10">
        <div className="flex flex-col items-start justify-between gap-8 sm:flex-row">
          <div>
            <div className="flex items-center gap-2 text-sm font-bold tracking-tight">
              <span className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-accent to-accent/60 text-[11px] text-white shadow-glow">
                ⚾
              </span>
              Diamond<span className="text-accent">Signal</span>
            </div>
            <p className="mt-3 max-w-xs text-xs leading-relaxed text-muted">
              A non-commercial research project. Predictions are model output, graded
              honestly against results — win or miss, it all shows up on the scorecard.
              Not betting advice.
            </p>
          </div>

          <div className="text-xs text-muted sm:text-right">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-fg/60">
              Data sources
            </div>
            <ul className="mt-2 flex max-w-xs flex-wrap gap-x-1.5 gap-y-1 sm:justify-end">
              {SOURCES.map((s, i) => (
                <li key={s}>
                  {s}
                  {i < SOURCES.length - 1 ? <span className="text-border"> ·</span> : null}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-8 flex flex-col items-start justify-between gap-3 border-t border-border/70 pt-5 text-xs text-muted sm:flex-row sm:items-center">
          <span>© {year} Diamond Signal</span>
          <a
            href="https://github.com/prajitchennupati"
            target="_blank"
            rel="noopener noreferrer"
            className="group inline-flex items-center gap-1.5 transition-colors duration-300 hover:text-accent"
          >
            Made by <span className="font-semibold text-fg group-hover:text-accent">Prajit Chennupati</span>
            <span className="transition-transform duration-300 ease-premium group-hover:translate-x-0.5">
              →
            </span>
          </a>
        </div>
      </div>
    </footer>
  );
}
