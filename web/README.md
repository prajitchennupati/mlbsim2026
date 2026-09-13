# web — MLBPlayoffs2026 frontend

Next.js 15 (App Router) · TypeScript · Tailwind · Recharts. Server Components fetch
from the FastAPI backend; the live board (`/`, `/live`) renders per-request and a
small client island re-pulls it on a timer, while the rest of the site uses ISR
and the pipeline pings `/api/revalidate` after each run.

## Develop

```bash
cp .env.example .env.local        # point API_URL at the running FastAPI service
npm install
npm run gen:types                 # optional: exact OpenAPI types -> lib/api-types.ts
npm run dev                       # http://localhost:3000
npm run lint && npm run typecheck
```

The backend must be running (`make api` in the repo root, default `:8000`).

## Pages

| Route | Source endpoint(s) |
|---|---|
| `/`, `/live` | `/games?date=` · `/predictions/summary` · `/predictions/history?resolved=true` · `/playoffs` |
| `/games`, `/games/[gamePk]` | `/games` · `/games/{pk}` + `/innings` `/players` `/simulation` · `/predictions/{id}/explanation` |
| `/predictions` | `/predictions/summary` · `/predictions/history` (filter: graded / pending / all, by model) |
| `/teams`, `/teams/[abbr]` | `/teams` · `/teams/{abbr}` + `/schedule` |
| `/standings` | `/standings` |
| `/playoffs` | `/playoffs` |
| `/model` | `/models` · `/models/{id}/evaluation` |
| `/players/[playerId]` | placeholder until `/players/{id}` lands |

Each game is predicted (win prob, projected score, batter/pitcher stat lines); once
it goes Final the pipeline's `resolve_outcomes` step grades it and the UI shows a
**✓ Correct / ✗ Miss** verdict plus the rolling accuracy / Brier / log-loss scorecard.

## Deploy (Vercel)

1. Import the repo; set **Root Directory** = `web/`. `vercel.json` pins the
   framework, build command, and region (`iad1`).
2. Environment variables (Production + Preview):

   | Var | Value |
   |---|---|
   | `API_URL` | the Render API URL, e.g. `https://mlbsim-api.onrender.com` |
   | `NEXT_PUBLIC_API_URL` | same URL |
   | `NEXT_PUBLIC_SITE_URL` | the Vercel URL, e.g. `https://mlbplayoffs2026.vercel.app` |
   | `REVALIDATE_SECRET` | a random string |

3. On the pipeline side (GitHub repo vars / Render env) set
   `MLBSIM_WEB_REVALIDATE_URL=https://<site>/api/revalidate` and
   `MLBSIM_REVALIDATE_SECRET` to the **same** value as `REVALIDATE_SECRET`.
4. The API's CORS already allows `https://*.vercel.app` when
   `MLBSIM_ENVIRONMENT=production`.

Once `web/package-lock.json` is committed, `vercel.json`'s `npm ci` path is used
automatically (it falls back to `npm install` until then).
