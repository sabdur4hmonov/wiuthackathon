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

## Phase 6 current checkpoint (2026-09-24)

The presentation layer now reads `/api/health` and reports the adapter as
available or unavailable/disconnected only from the API's explicit
`model_connected` value; an unavailable health response remains unknown.
Viewing fixture or catalog predictions is labelled as supplied data viewed
without live model execution. Empty events do not imply that the whole video
was checked, and an all-zero risk series is described as recorded model output,
not proof of safety or an accuracy claim. “Validated” sample wording means
prediction format/schema validation only, not detection-accuracy validation.

Phase 6 geometry was authored on a 3840×2160 frame at frame 3400 / 113.4467s.
The zones validator passed with zero errors, with a checkpoint inventory of 12
lanes, 1 stop line, 3 crossings, 1 queue zone, 1 signal, and 7 markings. This
does not validate detection accuracy. The protected backend geometry was not
modified by the website work; its current four queue zones remain outside the
website ownership boundary.

Part A stopped early because the Part B reserve estimate became approximately
3452s, making the Part A hard limit 0s. Actual Part A coverage was approximately
0.6s / 10 processed frames. Total observed runtime was 1003.2s / 340.34s =
2.9476x, with Part B taking 954.3s. The recorded risk output was all zero, and
an approximately 22 px camera-pose offset was observed during the first
approximately 30s. Full-video perception coverage was not verified. A
format-valid prediction is not a meaningful full-video evaluation, and no
accuracy claim is made without verified ground truth.
