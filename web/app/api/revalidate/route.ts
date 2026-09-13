import { revalidatePath, revalidateTag } from "next/cache";
import { NextResponse } from "next/server";

/**
 * On-demand revalidation webhook — the pipeline pings this after each run so
 * pages refresh without waiting for the ISR window. Accepts the secret + targets
 * either as query params or a JSON body:
 *
 *   POST /api/revalidate?secret=…&path=/games&tag=games
 *   POST /api/revalidate   { "secret": "…", "paths": ["/", "/games"], "tags": [] }
 */
const DEFAULT_PATHS = ["/", "/live", "/games", "/predictions", "/standings", "/playoffs", "/teams"];

export async function POST(request: Request) {
  const url = new URL(request.url);

  let body: { secret?: string; paths?: string[]; tags?: string[] } = {};
  try {
    body = (await request.json()) ?? {};
  } catch {
    /* no JSON body — fall back to query params */
  }

  const secret = url.searchParams.get("secret") ?? body.secret;
  if (!process.env.REVALIDATE_SECRET || secret !== process.env.REVALIDATE_SECRET) {
    return NextResponse.json({ ok: false, error: "bad secret" }, { status: 401 });
  }

  const paths = [...url.searchParams.getAll("path"), ...(body.paths ?? [])];
  const tags = [...url.searchParams.getAll("tag"), ...(body.tags ?? [])];
  const targets =
    paths.length || tags.length ? { paths, tags } : { paths: DEFAULT_PATHS, tags: [] as string[] };

  for (const p of targets.paths) revalidatePath(p);
  for (const t of targets.tags) revalidateTag(t);

  return NextResponse.json({ ok: true, revalidated: targets, at: Date.now() });
}
