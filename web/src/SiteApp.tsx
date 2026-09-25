import { useEffect, useMemo, useState } from "react";
import { AnnotatedPlayer } from "./components/AnnotatedPlayer.tsx";
import { EventTimeline } from "./components/EventTimeline.tsx";
import { RiskCurve } from "./components/RiskCurve.tsx";
import { ApproachSection, EdaSection, ReportSection, ResultsExtras } from "./SiteSections.tsx";
import { coverageMessage, executionMessage, FORMAT_VALIDATION_MESSAGE, type CoverageStatus } from "./lib/disclosures.ts";
import { parsePredictions, type ParsedPredictions, type VideoPrediction } from "./lib/predictions.ts";
import { loadSiteAssets, type RealSample, type SiteAssets } from "./lib/siteAssets.ts";
import { buildTimeline } from "./lib/timeline.ts";

const LIVE_DEMO_URL = "https://wiuthackathon-gerwwm75st8xkhkapprvc79.streamlit.app";
const navigation = [["home", "Home"], ["team", "Team"], ["approach", "Approach"], ["eda", "EDA"], ["results", "Results"], ["demo", "Live demo"], ["report", "Report"]] as const;
const FULL_SAMPLE_COVERAGE: CoverageStatus = { kind: "verified-full", detail: "fresh-clone, uncached full-clip harness output; the published risk curve is downsampled to 5 Hz" };

function sampleBundle(sample: RealSample): ParsedPredictions {
  const parsed = parsePredictions({ team: "wiut-cv", videos: { [sample.filename]: { events: sample.events, risk: sample.risk } } }, { kind: "sample", label: `Real model output from site_assets for ${sample.filename}` });
  if (!parsed.ok) throw new Error(parsed.errors.join("; "));
  return parsed.value;
}

function PredictionReview({ sample }: { sample: RealSample }) {
  const bundle = useMemo(() => sampleBundle(sample), [sample]);
  const prediction: VideoPrediction = bundle.document.videos[sample.filename];
  const [seekToSec, setSeekToSec] = useState<number | undefined>();
  const [seekToken, setSeekToken] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const timeline = buildTimeline(prediction.events, prediction.risk, sample.durationSec);
  function seek(seconds: number) { setSeekToSec(seconds); setSeekToken((token) => token + 1); setCurrentTime(seconds); }
  return <div className="review-stack">
    <div className="review-toolbar"><div><span className="eyebrow">site_assets model output</span><strong>{sample.filename}</strong></div><span className="filename-pill">{sample.durationSec.toFixed(2)} s</span></div>
    <div className="review-disclosures" aria-label="Prediction limitations"><p>{executionMessage(bundle.source.kind)}</p><p>{coverageMessage(FULL_SAMPLE_COVERAGE)}</p><p>{FORMAT_VALIDATION_MESSAGE}</p></div>
    <div className="review-grid">
      <AnnotatedPlayer filename={sample.filename} videoUrl={sample.annotatedVideoUrl} events={prediction.events} source={bundle.source} seekToSec={seekToSec} seekToken={seekToken} onTimeChange={setCurrentTime} />
      <EventTimeline prediction={prediction} source={bundle.source} durationHint={sample.durationSec} currentTime={currentTime} onSeek={seek} />
    </div>
    <RiskCurve prediction={prediction} source={bundle.source} durationSec={timeline.durationSec} currentTime={currentTime} onSeek={seek} />
  </div>;
}

function EvidenceLoading({ error }: { error: string | null }) {
  return <div className="placeholder-card"><span className="placeholder-icon">◇</span><div><h3>{error ? "Evidence could not be loaded" : "Loading measured evidence…"}</h3><p>{error ?? "Reading the committed site_assets package."}</p></div></div>;
}

export function SiteApp() {
  const [assets, setAssets] = useState<SiteAssets | null>(null);
  const [assetError, setAssetError] = useState<string | null>(null);
  const [selectedClip, setSelectedClip] = useState("sample_001");
  useEffect(() => {
    const controller = new AbortController();
    loadSiteAssets(controller.signal).then(setAssets).catch((error: unknown) => {
      if (!(error instanceof DOMException && error.name === "AbortError")) setAssetError(error instanceof Error ? error.message : "Unknown site_assets error.");
    });
    return () => controller.abort();
  }, []);
  const selectedSample = assets?.samples.find((sample) => sample.id === selectedClip) ?? assets?.samples[0];

  return <>
    <header className="site-header">
      <a className="brand" href="#home" aria-label="WIUT CV home"><span className="brand-mark"><span /></span><span>WIUT<span className="brand-accent">CV</span><small>INTELLIGENCE IN USE</small></span></a>
      <nav aria-label="Main navigation">{navigation.map(([id, label]) => <a href={`#${id}`} key={id}>{label}</a>)}</nav>
      <a className="header-link" href={LIVE_DEMO_URL} target="_blank" rel="noreferrer">Live demo ↗</a>
    </header>
    <main>
      <section className="hero" id="home"><div className="hero-grid" aria-hidden="true" /><div className="hero-copy"><span className="hero-kicker"><span className="live-dot" /> WIUT HACKATHON 2026 · COMPUTER VISION</span><h1>From traffic footage<br /><em>to actionable events.</em></h1><p>A budget-aware pipeline that turns fixed-camera footage into reviewable traffic-event intervals, annotated playback and a causal accident-risk stream.</p><div className="hero-actions"><a className="button primary" href={LIVE_DEMO_URL} target="_blank" rel="noreferrer">Open live demo <span>↗</span></a><a className="button secondary" href="#results">Inspect real results</a></div><p className="hero-disclosure">All evidence sections below load from the committed <code>site_assets/</code> package. No illustrative fixture is used.</p></div><div className="hero-visual" aria-hidden="true"><div className="road road-one" /><div className="road road-two" /><div className="scan-frame scan-a"><span>PERSON</span></div><div className="scan-frame scan-b"><span>VEHICLE</span></div><div className="crosshair">+</div><div className="visual-caption"><span>●</span> MODEL PIPELINE</div></div></section>

      <section className="section shell" id="team" data-source="site_assets">
        <div className="section-heading"><span className="eyebrow">01 / The team</span><h2>{assets ? `Meet Team ${assets.team.name}.` : "The people behind the pipeline."}</h2><p>Member names and links are loaded from <code>site_assets/team.json</code>, sourced from the project README.</p></div>
        {assets ? <>
          <div className="process-grid team-workstreams">{assets.team.members.map((member, index) => <article className="process-card" key={member.name}>
            <span>{String(index + 1).padStart(2, "0")}</span><h3>{member.name}</h3>
            {member.role && <p>{member.role}</p>}
            <div className="team-links">{member.links.map((link) => <a href={link.url} target="_blank" rel="noreferrer" key={link.url}>{link.label} ↗</a>)}</div>
          </article>)}</div>
          <p className="evidence-note">{assets.team.attribution}</p>
        </> : <EvidenceLoading error={assetError} />}
      </section>

      <ApproachSection />

      <EdaSection assets={assets} error={assetError} />

      <section className="section shell" id="results" data-source="site_assets"><div className="section-heading"><span className="eyebrow">04 / Results</span><h2>Real harness output, synchronized.</h2><p>Events, 5 Hz risk curves and annotated videos load directly from <code>site_assets/events</code>, <code>risk</code> and <code>videos</code>.</p></div>{!assets || !selectedSample ? <EvidenceLoading error={assetError} /> : <><div className="clip-tabs">{assets.samples.map((sample) => <button className={`button secondary${sample.id === selectedSample.id ? " active" : ""}`} key={sample.id} onClick={() => setSelectedClip(sample.id)}>{sample.id}</button>)}</div><PredictionReview key={selectedSample.id} sample={selectedSample} /><div className="failure-grid">{assets.failures.filter((failure) => ["moped_rider", "far_kerb_pedestrians", "truck_queue"].includes(failure.id)).map((failure) => <article key={failure.id}><video src={failure.videoUrl} controls preload="none" /><span className="eyebrow">Failure case · {failure.clip}</span><h3>{failure.title}</h3><p>{failure.what}</p><p className="failure-status"><strong>Status:</strong> {failure.status}</p></article>)}</div><ResultsExtras assets={assets} /></>}</section>

      <section className="section shell" id="demo"><div className="demo-callout"><div><span className="eyebrow">05 / Live inference</span><h2>Try the deployed model.</h2><p>Upload a road-camera MP4 to the confirmed Streamlit deployment to receive event intervals, annotated playback and a risk curve.</p><p className="small-note">Streamlit Community Cloud runs on CPU (about 2–4× clip duration). Uploads are limited to 120 seconds / 200 MB. Event zones are calibrated for this intersection; other camera views may produce detections without event rules.</p></div><a className="button primary" href={LIVE_DEMO_URL} target="_blank" rel="noreferrer">Launch live demo ↗</a></div></section>

      <ReportSection assets={assets} error={assetError} />
    </main>
    <footer className="site-footer shell"><span>WIUT CV · Intelligence in Use</span><span>Evidence source: site_assets/ · Live demo: Streamlit</span></footer>
  </>;
}
