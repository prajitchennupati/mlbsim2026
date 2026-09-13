import "server-only";

const BASE =
  process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const API_BASE = BASE;

export class ApiError extends Error {
  constructor(
    public status: number,
    public path: string,
    message: string,
  ) {
    super(message);
  }
}

interface Opts {
  /** ISR revalidation window in seconds (default 5 min). Use 0 for no-store. */
  revalidate?: number;
  tags?: string[];
  /**
   * Fail the request after this long instead of hanging (default 45s — the
   * free-tier Render API can take 30-50s to wake from a cold sleep, and a
   * fetch with no bound at all risks the whole Vercel function getting
   * killed by the platform instead of failing into apiGetOrNull/Default's
   * catch, which is what shows the generic error boundary).
   */
  timeoutMs?: number;
}

export async function apiGet<T>(path: string, opts: Opts = {}): Promise<T> {
  const url = `${BASE}/api/v1${path}`;
  const revalidate = opts.revalidate ?? 300;
  const init: RequestInit & { next?: { revalidate?: number; tags?: string[] } } =
    revalidate === 0
      ? { cache: "no-store" }
      : { next: { revalidate, tags: opts.tags } };
  const res = await fetch(url, {
    headers: { accept: "application/json" },
    signal: AbortSignal.timeout(opts.timeoutMs ?? 45_000),
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json())?.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, path, detail);
  }
  return (await res.json()) as T;
}

/**
 * Like apiGet but returns null instead of throwing — on a 404, and on transient
 * failures (network, 5xx) so a page can degrade to an empty state and, during a
 * backend-less build, still prerender.
 */
export async function apiGetOrNull<T>(path: string, opts?: Opts): Promise<T | null> {
  try {
    return await apiGet<T>(path, opts);
  } catch (e) {
    warnOnFailure(path, e);
    return null;
  }
}

/**
 * Like apiGetOrNull, but returns `fallback` (typically `[]`) instead of null —
 * for list endpoints a page wants to render as empty rather than special-case.
 * Always logs, unlike a bare `.catch(() => [])`, so a misconfigured API_URL
 * shows up in the deployment's function logs instead of silently rendering an
 * empty page with no trace of why.
 */
export async function apiGetOrDefault<T>(path: string, fallback: T, opts?: Opts): Promise<T> {
  try {
    return await apiGet<T>(path, opts);
  } catch (e) {
    warnOnFailure(path, e);
    return fallback;
  }
}

function warnOnFailure(path: string, e: unknown): void {
  if (e instanceof ApiError && e.status !== 404) {
    console.warn(`API ${path} -> ${e.status} ${e.message}`);
  } else if (!(e instanceof ApiError)) {
    console.warn(`API ${path} -> unreachable (${API_BASE}): ${(e as Error).message}`);
  }
}
