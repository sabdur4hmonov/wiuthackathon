# WIUT CV website

React + TypeScript + Vite presentation layer. Install with `npm ci`; use
`npm run dev`, `npm test`, and `npm run build` from this directory. The local
development server proxies `/api` to `http://127.0.0.1:8765`.

`src/lib/predictions.ts` validates the organizer JSON and tracks provenance
outside it. `src/components/` contains the event timeline, risk curve and
optional annotated player. `src/fixtures/` contains only explicitly
illustrative interface data. The page intentionally leaves real EDA, sample
results, team identities and performance claims pending.

See [the integration plan](../INTEGRATION.md) for the model adapter seam and
data/limitation details.
