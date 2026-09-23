import { useEffect, useMemo, useState } from "react";
import { AnnotatedPlayer } from "./components/AnnotatedPlayer.tsx";
import { EventTimeline } from "./components/EventTimeline.tsx";
import { RiskCurve } from "./components/RiskCurve.tsx";
import illustrativeJson from "./fixtures/illustrative_predictions.json";
import { createJob, getJob, getJobResult, type JobView } from "./lib/api.ts";
import {
  parsePredictions,
  type ParsedPredictions,
  type PredictionSource,
  type VideoPrediction,
} from "./lib/predictions.ts";
import { buildTimeline } from "./lib/timeline.ts";

const fixtureSource: PredictionSource = {
  kind: "fixture",
  label: "Invented values for interface testing; not camera footage or model output",
};
function loadFixture(): ParsedPredictions {
  const fixture = parsePredictions(illustrativeJson, fixtureSource);
  if (!fixture.ok) throw new Error(`Invalid development fixture: ${fixture.errors.join("; ")}`);
  return fixture.value;
}
const illustrativeFixture = loadFixture();

const navigation = [
  ["home", "Home"],
  ["team", "Team"],
  ["approach", "Approach"],
  ["eda", "EDA"],
  ["results", "Results"],
  ["demo", "Live demo"],
  ["report", "Report"],
] as const;

function PredictionReview({
  bundle,
  videoUrl,
}: {
  bundle: ParsedPredictions;
  videoUrl?: string;
}) {
  const filenames = Object.keys(bundle.document.videos);
  const [filename, setFilename] = useState(filenames[0] ?? "");
  const [seekToSec, setSeekToSec] = useState<number | undefined>();
  const [seekToken, setSeekToken] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const selectedName = filenames.includes(filename) ? filename : filenames[0];
  const prediction: VideoPrediction | undefined = bundle.document.videos[selectedName];

  function seek(seconds: number) {
    setSeekToSec(seconds);
    setSeekToken((token) => token + 1);
    setCurrentTime(seconds);
  }

  if (!prediction) return <div className="empty-state">No videos in this prediction file.</div>;
  const timeline = buildTimeline(prediction.events, prediction.risk);
  return (
    <div className="review-stack">
      <div className="review-toolbar">
        <div>
          <span className="eyebrow">Prediction document</span>
          <strong>{bundle.document.team ?? "Unnamed team"}</strong>
        </div>
        {filenames.length > 1 ? (
          <label>
            Video
            <select value={selectedName} onChange={(event) => setFilename(event.target.value)}>
              {filenames.map((name) => <option key={name}>{name}</option>)}
            </select>
          </label>
        ) : (
          <span className="filename-pill">{selectedName}</span>
        )}
      </div>
      <div className="review-grid">
        <AnnotatedPlayer
          filename={selectedName}
          videoUrl={videoUrl}
          events={prediction.events}
          source={bundle.source}
          seekToSec={seekToSec}
          seekToken={seekToken}
          onTimeChange={setCurrentTime}
        />
        <EventTimeline
          prediction={prediction}
          source={bundle.source}
          durationHint={timeline.durationSec}
          currentTime={currentTime}
          onSeek={seek}
        />
      </div>
      <RiskCurve
        prediction={prediction}
        source={bundle.source}
        durationSec={timeline.durationSec}
        currentTime={currentTime}
        onSeek={seek}
      />
      {bundle.warnings.length > 0 && (
        <p className="small-note">Parser notes: {bundle.warnings.join(" ")}</p>
      )}
    </div>
  );
}

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<JobView | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadResult, setUploadResult] = useState<ParsedPredictions | null>(null);
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : undefined), [file]);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(() => {
      getJob(job.job_id)
        .then(async (nextJob) => {
          setJob(nextJob);
          setUploadError(null);
          if (nextJob.status === "completed") {
            const raw = await getJobResult(nextJob.job_id);
            const parsed = parsePredictions(raw, {
              kind: "upload",
              label: `Model output for ${nextJob.filename}`,
            });
            if (!parsed.ok) throw new Error(`Invalid model result: ${parsed.errors.join("; ")}`);
            setUploadResult(parsed.value);
          }
        })
        .catch((error: unknown) => setUploadError(String(error)));
    }, 1200);
    return () => window.clearInterval(timer);
  }, [job]);

  async function upload() {
    if (!file) return;
    setUploadBusy(true);
    setUploadError(null);
    setUploadResult(null);
    setJob(null);
    try {
      if (!file.name.toLowerCase().endsWith(".mp4")) throw new Error("Select an .mp4 file.");
      if (file.size > 100 * 1024 * 1024) throw new Error("This foundation demo accepts files up to 100 MiB.");
      setJob(await createJob(file));
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : String(error));
    } finally {
      setUploadBusy(false);
    }
  }

  return (
    <>
      <header className="site-header">
        <a className="brand" href="#home" aria-label="WIUT CV home">
          <span className="brand-mark"><span /></span>
          <span>WIUT<span className="brand-accent">CV</span><small>INTELLIGENCE IN USE</small></span>
        </a>
        <nav aria-label="Main navigation">
          {navigation.map(([id, label]) => <a href={`#${id}`} key={id}>{label}</a>)}
        </nav>
        <a className="header-link" href="https://github.com/sabdur4hmonov/wiuthackathon" target="_blank" rel="noreferrer">
          Repository ↗
        </a>
      </header>

      <main>
        <section className="hero" id="home">
          <div className="hero-grid" aria-hidden="true" />
          <div className="hero-copy">
            <span className="hero-kicker"><span className="live-dot" /> WIUT HACKATHON 2026 · COMPUTER VISION</span>
            <h1>From traffic footage<br /><em>to actionable events.</em></h1>
            <p>
              A fixed-camera traffic intelligence project: detect event intervals and estimate
              accident risk before impact. This website currently presents the integration foundation.
            </p>
            <div className="hero-actions">
              <a className="button primary" href="#demo">Explore the demo</a>
              <a className="button secondary" href="#approach">See the approach <span>↗</span></a>
            </div>
            <p className="hero-disclosure">Real camera results and measured performance will appear only after model validation.</p>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="road road-one" /><div className="road road-two" />
            <div className="scan-frame scan-a"><span>ZONE 01</span></div>
            <div className="scan-frame scan-b"><span>TRACK / PENDING</span></div>
            <div className="crosshair">+</div>
            <div className="visual-caption">CAMERA 01 <span>•</span> SINGLE FIXED VIEW</div>
          </div>
        </section>

        <section className="section shell" id="team">
          <div className="section-heading"><span className="eyebrow">01 / The people</span><h2>Built by a team of three.</h2></div>
          <div className="placeholder-card team-placeholder">
            <span className="placeholder-icon">◎</span>
            <div><h3>Team profiles pending</h3><p>Verified names, roles, contributions, portfolios and social links will be added here. No team member details are invented.</p></div>
          </div>
        </section>

        <section className="section shell" id="approach">
          <div className="section-heading"><span className="eyebrow">02 / The challenge</span><h2>One camera. Many possible events.</h2><p>Each prediction is a time interval with one official label. The optional risk stream scores each frame for an accident beginning within five seconds.</p></div>
          <div className="process-grid">
            <article className="process-card"><span>01</span><h3>Video input</h3><p>One MP4 from the fixed road camera.</p></article>
            <article className="process-card"><span>02</span><h3>Model contract</h3><p><code>detect_events()</code> and causal <code>RiskEstimator.step()</code>.</p></article>
            <article className="process-card"><span>03</span><h3>Official harness</h3><p>Unchanged runner writes the canonical prediction JSON.</p></article>
            <article className="process-card"><span>04</span><h3>Operator view</h3><p>Timeline, risk curve, synchronized playback and review.</p></article>
          </div>
          <div className="info-strip"><strong>Model status</strong><span>Scene zones and event rules are still being developed. This page makes no detection-quality claim.</span></div>
        </section>

        <section className="section shell" id="eda">
          <div className="section-heading"><span className="eyebrow">03 / Exploratory analysis</span><h2>Evidence before interpretation.</h2><p>The real sample videos have not been processed into publishable EDA in this repository.</p></div>
          <div className="placeholder-card"><span className="placeholder-icon">▦</span><div><h3>Sample analysis awaiting source material</h3><p>Resolution, FPS, duration, object counts, motion heatmaps, trajectories, lane directions, density and failure cases will be reported from actual footage and measurements.</p></div></div>
        </section>

        <section className="section shell" id="results">
          <div className="section-heading"><span className="eyebrow">04 / Results</span><h2>Measured results belong here.</h2><p>Real sample timelines, annotated videos and performance analysis will be published after inference on the organizer clips and manual validation.</p></div>
          <div className="placeholder-card"><span className="placeholder-icon">◇</span><div><h3>No real sample results yet</h3><p>The repository's current <code>predictions_samples.json</code> is from a generated synthetic clip and is excluded from this results section.</p></div></div>
        </section>

        <section className="section shell" id="demo">
          <div className="section-heading"><span className="eyebrow">05 / Interface laboratory</span><h2>Explore the prediction contract.</h2><p>The review below uses explicitly illustrative numbers to exercise the interface. They are not model detections.</p></div>
          <div className="fixture-banner"><strong>ILLUSTRATIVE FIXTURE</strong><span>Invented intervals and scores · no camera video · no measured performance</span></div>
          <PredictionReview bundle={illustrativeFixture} />

          <div className="upload-block">
            <div><span className="eyebrow">Upload foundation</span><h3>Bring your own MP4</h3><p>The local API accepts MP4 uploads up to 100 MiB and 120 declared seconds. Jobs expire after 15 minutes. Model invocation is intentionally disconnected in this phase; accepted jobs report that state plainly.</p></div>
            <div className="upload-controls">
              <label className="file-label">
                <span>{file?.name ?? "Choose an .mp4 file"}</span>
                <input type="file" accept="video/mp4,.mp4" onChange={(event) => {
                  setFile(event.target.files?.[0] ?? null);
                  setJob(null);
                  setUploadResult(null);
                  setUploadError(null);
                }} />
              </label>
              <button className="button primary" disabled={!file || uploadBusy} onClick={upload}>
                {uploadBusy ? "Uploading…" : "Create demo job"}
              </button>
            </div>
            {uploadError && <p className="feedback error" role="alert">{uploadError}</p>}
            {job && <p className="feedback" role="status"><strong>{job.status.replaceAll("_", " ")}</strong> · {job.message} · Job {job.job_id.slice(0, 8)}</p>}
            {uploadResult && <PredictionReview bundle={uploadResult} videoUrl={previewUrl} />}
          </div>
        </section>

        <section className="section shell" id="report">
          <div className="section-heading"><span className="eyebrow">06 / Technical report</span><h2>What is built, and what remains.</h2></div>
          <div className="report-grid">
            <article><span className="report-marker done">✓</span><h3>Available now</h3><p>Typed prediction validation, source-labelled visualizations, synchronized playback interface and an upload job API that keeps server paths private.</p></article>
            <article><span className="report-marker pending">→</span><h3>Awaiting model data</h3><p>Real event intervals, risk curves, camera EDA, annotated sample videos and measured failure analysis.</p></article>
            <article><span className="report-marker pending">→</span><h3>Next integration</h3><p>Connect the unchanged organizer harness through the demo adapter, then test real uploads and deployment limits.</p></article>
          </div>
          <div className="resource-row"><a href="https://github.com/sabdur4hmonov/wiuthackathon" target="_blank" rel="noreferrer">Source repository ↗</a><span>Weights and real sample prediction links will be added after verification.</span></div>
        </section>
      </main>

      <footer className="site-footer shell"><span>WIUT CV · Intelligence in Use</span><span>Built for WIUT Hackathon 2026</span></footer>
    </>
  );
}
