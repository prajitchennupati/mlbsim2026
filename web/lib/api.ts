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
    if (e instanceof ApiError && e.status !== 404) {
      console.warn(`apiGetOrNull ${path} -> ${e.status} ${e.message}`);
    } else if (!(e instanceof ApiError)) {
      console.warn(`apiGetOrNull ${path} -> ${(e as Error).message}`);
    }
    return null;
  }
}
