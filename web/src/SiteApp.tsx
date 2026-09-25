import { useEffect, useMemo, useState } from "react";
import { AnnotatedPlayer } from "./components/AnnotatedPlayer.tsx";
import { EventTimeline } from "./components/EventTimeline.tsx";
import { RiskCurve } from "./components/RiskCurve.tsx";
import { coverageMessage, executionMessage, FORMAT_VALIDATION_MESSAGE, type CoverageStatus } from "./lib/disclosures.ts";
import { parsePredictions, type ParsedPredictions, type VideoPrediction } from "./lib/predictions.ts";
import { loadSiteAssets, type RealSample, type SiteAssets } from "./lib/siteAssets.ts";
import { buildTimeline } from "./lib/timeline.ts";

const LIVE_DEMO_URL = "https://wiuthackathon-gerwwm75st8xkhkapprvc79.streamlit.app";
const SOURCE_URL = "https://github.com/sabdur4hmonov/wiuthackathon";
const integer = new Intl.NumberFormat("en-US");
const navigation = [["home", "Home"], ["team", "Team"], ["approach", "Approach"], ["eda", "EDA"], ["results", "Results"], ["demo", "Live demo"], ["report", "Report"]] as const;
const FULL_SAMPLE_COVERAGE: CoverageStatus = { kind: "verified-full", detail: "complete Stage 1 cache and full-clip harness output; the published risk curve is downsampled to 5 Hz" };

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
  const totalTracks = assets?.eda.reduce((sum, clip) => sum + clip.tracks, 0) ?? 0;

  return <>
    <header className="site-header">
      <a className="brand" href="#home" aria-label="WIUT CV home"><span className="brand-mark"><span /></span><span>WIUT<span className="brand-accent">CV</span><small>INTELLIGENCE IN USE</small></span></a>
      <nav aria-label="Main navigation">{navigation.map(([id, label]) => <a href={`#${id}`} key={id}>{label}</a>)}</nav>
      <a className="header-link" href={LIVE_DEMO_URL} target="_blank" rel="noreferrer">Live demo ↗</a>
    </header>
    <main>
      <section className="hero" id="home"><div className="hero-grid" aria-hidden="true" /><div className="hero-copy"><span className="hero-kicker"><span className="live-dot" /> WIUT HACKATHON 2026 · COMPUTER VISION</span><h1>From traffic footage<br /><em>to actionable events.</em></h1><p>A budget-aware pipeline that turns fixed-camera footage into reviewable traffic-event intervals, annotated playback and a causal accident-risk stream.</p><div className="hero-actions"><a className="button primary" href={LIVE_DEMO_URL} target="_blank" rel="noreferrer">Open live demo <span>↗</span></a><a className="button secondary" href="#results">Inspect real results</a></div><p className="hero-disclosure">All evidence sections below load from the committed <code>site_assets/</code> package. No illustrative fixture is used.</p></div><div className="hero-visual" aria-hidden="true"><div className="road road-one" /><div className="road road-two" /><div className="scan-frame scan-a"><span>PERSON · 0.91</span></div><div className="scan-frame scan-b"><span>VEHICLE · 0.96</span></div><div className="crosshair">+</div><div className="visual-caption"><span>●</span> REAL SAMPLE EVIDENCE</div></div></section>

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

      <section className="section shell" id="approach"><div className="section-heading"><span className="eyebrow">02 / The challenge</span><h2>One camera. A strict 3× budget.</h2><p>Part A detects temporal events; Part B scores whether an accident may begin within five seconds. Both share one runtime budget.</p></div><div className="process-grid"><article className="process-card"><span>01</span><h3>Keyframe decode</h3><p>Fixed 15-frame GOPs make a 2 Hz perception pass practical.</p></article><article className="process-card"><span>02</span><h3>Detect + track</h3><p>YOLO11s and motion-tolerant ByteTrack produce scene tracks.</p></article><article className="process-card"><span>03</span><h3>Scene rules</h3><p>Aligned lanes, crossings and queue zones drive event logic.</p></article><article className="process-card"><span>04</span><h3>Causal risk</h3><p>A separate online estimator sees only frames already observed.</p></article></div></section>

      <section className="section shell" id="eda" data-source="site_assets"><div className="section-heading"><span className="eyebrow">03 / Exploratory analysis</span><h2>Measured on all four real clips.</h2><p>Counts, trajectories, camera pose and codec findings are loaded from <code>site_assets/eda/</code>.</p></div>{!assets ? <EvidenceLoading error={assetError} /> : <><div className="eda-totals"><article><strong>{assets.eda.length}</strong><span>real 4K clips</span></article><article><strong>{integer.format(totalTracks)}</strong><span>track IDs</span></article><article><strong>{assets.format.gop.length_frames}</strong><span>frames / GOP</span></article><article><strong>{assets.format.gop.length_sec.toFixed(3)} s</strong><span>keyframe interval</span></article><article><strong>{assets.format.footage.fps.toFixed(2)}</strong><span>frames / second</span></article><article><strong>{assets.format.footage.bitrate_mbps}</strong><span>Mb/s footage</span></article></div><div className="eda-table-wrap"><table className="eda-table"><thead><tr><th>Clip</th><th>Duration</th><th>Tracks</th><th>Detections / sample</th><th>People</th><th>Cars</th><th>Alignment drift</th></tr></thead><tbody>{assets.eda.map((clip) => <tr key={clip.clip}><th>{clip.clip}</th><td>{clip.duration_sec.toFixed(1)} s</td><td>{integer.format(clip.tracks)}</td><td>{clip.detections_per_sample.toFixed(1)}</td><td>{integer.format(clip.by_class_tracks.person ?? 0)}</td><td>{integer.format(clip.by_class_tracks.car ?? 0)}</td><td>{clip.pose.drift_px.toFixed(2)} px</td></tr>)}</tbody></table></div><div className="eda-visuals">{assets.samples.map((sample) => <article key={sample.id}><h3>{sample.id}</h3><img src={sample.heatmapUrl} alt={`${sample.id} motion heatmap from site_assets`} loading="lazy" /><img src={sample.trajectoriesUrl} alt={`${sample.id} tracked trajectories and lanes from site_assets`} loading="lazy" /></article>)}</div><p className="evidence-note">{assets.format.footage.container} · {assets.format.footage.codec}. These are model-derived observations, not ground-truth accuracy measurements.</p></>}</section>

      <section className="section shell" id="results" data-source="site_assets"><div className="section-heading"><span className="eyebrow">04 / Results</span><h2>Real harness output, synchronized.</h2><p>Events, 5 Hz risk curves and annotated videos load directly from <code>site_assets/events</code>, <code>risk</code> and <code>videos</code>.</p></div>{!assets || !selectedSample ? <EvidenceLoading error={assetError} /> : <><div className="clip-tabs">{assets.samples.map((sample) => <button className={`button secondary${sample.id === selectedSample.id ? " active" : ""}`} key={sample.id} onClick={() => setSelectedClip(sample.id)}>{sample.id}</button>)}</div><PredictionReview key={selectedSample.id} sample={selectedSample} /><div className="failure-grid">{assets.failures.map((failure) => <article key={failure.id}><video src={failure.videoUrl} controls preload="metadata" /><span className="eyebrow">Failure case · {failure.clip}</span><h3>{failure.title}</h3><p>{failure.what}</p><p className="failure-status"><strong>Status:</strong> {failure.status}</p></article>)}</div></>}</section>

      <section className="section shell" id="demo"><div className="demo-callout"><div><span className="eyebrow">05 / Live inference</span><h2>Try the deployed model.</h2><p>Upload a road-camera MP4 to the confirmed Streamlit deployment to receive event intervals, annotated playback and a risk curve.</p><p className="small-note">Streamlit Community Cloud runs on CPU; clips are limited to 120 seconds / 200 MB.</p></div><a className="button primary" href={LIVE_DEMO_URL} target="_blank" rel="noreferrer">Launch live demo ↗</a></div></section>

      <section className="section shell" id="report" data-source="site_assets"><div className="section-heading"><span className="eyebrow">06 / Technical report</span><h2>{assets ? `${assets.team.name}: findings that changed the system.` : "Findings that changed the system."}</h2><p>The report cards below are rendered from <code>site_assets/eda/format_findings.json</code>, with limitations from <code>site_assets/failures/failures.json</code>.</p></div>{!assets ? <EvidenceLoading error={assetError} /> : <><div className="decision-grid">{assets.format.decisions.map((item, index) => <article key={item.finding}><span>{String(index + 1).padStart(2, "0")}</span><h3>{item.finding}</h3><p>{item.decision}</p></article>)}</div><p className="evidence-note">{assets.team.attribution}</p><div className="resource-row" id="links"><a href={SOURCE_URL} target="_blank" rel="noreferrer">Source repository ↗</a><a href={LIVE_DEMO_URL} target="_blank" rel="noreferrer">Live demo ↗</a><span>{assets.format.measured_costs_x_realtime.budget}</span></div></>}</section>
    </main>
    <footer className="site-footer shell"><span>WIUT CV · Intelligence in Use</span><span>Evidence source: site_assets/ · Live demo: Streamlit</span></footer>
  </>;
}
