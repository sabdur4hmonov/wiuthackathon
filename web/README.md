# WIUT CV website

React + TypeScript + Vite presentation layer. Install with `npm ci`; use
`npm run dev`, `npm test`, and `npm run build` from this directory. The local
development server proxies `/api` to `http://127.0.0.1:8765`.

`src/lib/predictions.ts` validates the organizer JSON and tracks provenance
outside it. `src/components/` contains the event timeline, risk curve and
optional annotated player. `src/fixtures/` contains only explicitly
illustrative interface data. `public/samples/catalog.json` is intentionally
empty until a real, authorized, sanitized sample is published. Upload runs
retain their submitted file independently of later selection; browser media
metadata sets timeline duration when available. The page leaves real EDA,
sample results, team identities and performance claims pending.

See [the integration plan](../INTEGRATION.md) for the model adapter seam and
data/limitation details.
