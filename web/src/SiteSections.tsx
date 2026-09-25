import { EdaChart } from "./components/EdaChart.tsx";
import { siteAssetUrl, type SiteAssets } from "./lib/siteAssets.ts";

const repo = "https://github.com/sabdur4hmonov/wiuthackathon";
const tag = `${repo}/tree/v1.0-final2`;
const integer = new Intl.NumberFormat("en-US");
const meanF1: Record<string, number> = { jaywalking: 0.551, stopped_vehicle: 0.303, congestion: 0 };

export function ApproachSection() {
  return <section className="section shell" id="approach">
    <div className="section-heading"><span className="eyebrow">02 / The challenge</span><h2>One camera. A strict 3× budget.</h2><p>Part A returns event intervals; Part B emits a causal accident-risk value for each frame. Both share the same runtime budget, and an overrun empties both outputs.</p></div>
    <div className="process-grid">
      <article className="process-card"><span>01 · DATA</span><h3>Keyframe decode</h3><p>PyAV reads one I-frame every 15 frames: about 2 samples per second on these clips.</p></article>
      <article className="process-card"><span>02 · LEARNED</span><h3>Detect objects</h3><p>COCO-pretrained YOLO11s finds people and vehicles. No hackathon sample was used for model training.</p></article>
      <article className="process-card"><span>03 · RULES</span><h3>Track and classify</h3><p>Motion-aware ByteTrack, camera-pose alignment, authored zones and event rules turn detections into intervals.</p></article>
      <article className="process-card"><span>04 · ANALYTIC</span><h3>Estimate risk</h3><p>A separate 5 Hz detector and time-to-collision estimator use only frames observed so far, with a runtime guard.</p></article>
    </div>
    <div className="approach-details">
      <article><h3>What is learned?</h3><p>YOLO11s is the only learned model. Track association, scene geometry, jaywalking, stopped-vehicle and congestion decisions, post-processing and the time-to-collision score are engineered rules. Wrong-way detection is gated off at the 0.5 s sample interval because car platoons alias.</p></article>
      <article><h3>Models, data and licences</h3><p><a href="https://docs.ultralytics.com/models/yolo11/" target="_blank" rel="noreferrer">Ultralytics YOLO11s</a> weights and implementation: AGPL-3.0. COCO is the pretraining dataset; its annotations are CC BY 4.0 and source-image licences vary. We do not redistribute COCO or retrain. PyAV/FFmpeg, OpenCV, NumPy, SciPy and lap are credited in the <a href={`${repo}/blob/v1.0-final2/README.md#models-data-and-licences`} target="_blank" rel="noreferrer">repository licence table</a>. The organizers' sample MP4s are not in the repository; this site shows our web-sized annotated outputs.</p></article>
    </div>
  </section>;
}

export function EdaSection({ assets, error }: { assets: SiteAssets | null; error: string | null }) {
  const totalTracks = assets?.eda.reduce((sum, clip) => sum + clip.tracks, 0) ?? 0;
  return <section className="section shell" id="eda" data-source="site_assets">
    <div className="section-heading"><span className="eyebrow">03 / Exploratory analysis</span><h2>Measured on four real clips.</h2><p>Camera format, object counts, lane occupancy, motion heatmaps and trajectories come from the committed sample analysis.</p></div>
    {!assets ? <p className="evidence-note">{error ?? "Loading measured evidence…"}</p> : <>
      <div className="eda-totals">
        <article><strong>{assets.eda.length}</strong><span>4K clips</span></article>
        <article><strong>{integer.format(totalTracks)}</strong><span>track IDs</span></article>
        <article><strong>{assets.format.gop.length_frames}</strong><span>frames / GOP</span></article>
        <article><strong>{assets.format.gop.length_sec.toFixed(3)} s</strong><span>keyframe interval</span></article>
        <article><strong>{assets.format.footage.fps.toFixed(2)}</strong><span>frames / second</span></article>
        <article><strong>{assets.format.footage.bitrate_mbps}</strong><span>Mb/s footage</span></article>
      </div>
      <div className="eda-table-wrap"><table className="eda-table"><thead><tr><th>Clip</th><th>Duration</th><th>Resolution</th><th>Tracks</th><th>Detections / sample</th><th>People</th><th>Cars</th><th>Alignment drift</th></tr></thead><tbody>{assets.eda.map((clip) => <tr key={clip.clip}><th>{clip.clip}</th><td>{clip.duration_sec.toFixed(1)} s</td><td>{assets.format.footage.resolution}</td><td>{integer.format(clip.tracks)}</td><td>{clip.detections_per_sample.toFixed(1)}</td><td>{integer.format(clip.by_class_tracks.person ?? 0)}</td><td>{integer.format(clip.by_class_tracks.car ?? 0)}</td><td>{clip.pose.drift_px.toFixed(1)} px</td></tr>)}</tbody></table></div>
      <p className="evidence-note">{assets.format.footage.container} · {assets.format.footage.codec}. The published frames show daylight with strong tree shade at the far kerb. This is a qualitative observation, not a measured lighting classification.</p>
      {assets.samples.map((sample, index) => <details className="eda-clip" key={sample.id} open={index === 0}>
        <summary>{sample.id} · object and motion evidence</summary>
        <div className="eda-charts">
          <EdaChart title="Tracked objects by class over time" data={sample.counts} note="Counts of tracked boxes at each 0.5 s Stage 1 sample; detections can miss small or occluded objects." />
          <EdaChart title="Vehicles in directional lanes over time" data={sample.density} note="Model-derived lane occupancy counts, not physical density or traffic volume." />
        </div>
        <div className="eda-visuals"><article><h3>Motion heatmap</h3><img src={sample.heatmapUrl} alt={`${sample.id} measured motion heatmap on a video frame`} loading="lazy" /></article><article><h3>Tracks and lane directions</h3><img src={sample.trajectoriesUrl} alt={`${sample.id} tracked paths and authored lane arrows`} loading="lazy" /></article></div>
      </details>)}
      <p className="evidence-note">The 4K, 10-bit H.264 video takes 1.2–3.0× realtime to decode in the harness on the development laptop. Its fixed 15-frame GOP made keyframe-only Part A practical; camera re-framing required per-clip zone alignment. Counts and tracks are model outputs, not ground truth.</p>
    </>}
  </section>;
}

export function ResultsExtras({ assets }: { assets: SiteAssets }) {
  const examples = [
    { label: "Jaywalking", clip: "sample_001 · 109.7 s", file: "sample_001_event_01_jaywalking_109.7s.mp4" },
    { label: "Stopped vehicle", clip: "sample_003 · 49.6 s", file: "sample_003_event_02_stopped_vehicle_49.6s.mp4" },
  ];
  return <>
    <h3 className="subsection-heading">Detected-class examples</h3>
    <div className="example-grid">{examples.map((example) => <article key={example.file}><video controls preload="none" src={siteAssetUrl(`event_clips/${example.file}`)} /><h4>{example.label}</h4><p>{example.clip} · annotated sample output</p></article>)}</div>
    <h3 className="subsection-heading">Error analysis on development labels</h3>
    <p className="evidence-note">Score A = 0.2846. The team made these labels by reviewing its own candidates, so the evaluation is circular and likely flattering. Counts below use tIoU ≥ {assets.errors.tiou_threshold.toFixed(1)}. Mean F1 averages tIoU 0.3, 0.5 and 0.7.</p>
    <div className="eda-table-wrap"><table className="eda-table"><thead><tr><th>Class</th><th>TP</th><th>FP</th><th>FN</th><th>F1 @ 0.5</th><th>Mean F1</th></tr></thead><tbody>{Object.entries(assets.errors.per_class).map(([name, value]) => <tr key={name}><th>{name.replaceAll("_", " ")}</th><td>{value.tp}</td><td>{value.fp}</td><td>{value.fn}</td><td>{value.f1.toFixed(3)}</td><td>{(meanF1[name] ?? 0).toFixed(3)}</td></tr>)}</tbody></table></div>
    <p className="small-note">No sample contains an accident, so the Part B risk curve has no validated accident-anticipation score. The one ambiguous truck-queue congestion label is missed.</p>
  </>;
}

export function ReportSection({ assets, error }: { assets: SiteAssets | null; error: string | null }) {
  return <section className="section shell" id="report" data-source="site_assets">
    <div className="section-heading"><span className="eyebrow">06 / Technical report</span><h2>What the samples taught us.</h2><p>Measured results, limits and next steps for Team Armagedon.</p></div>
    {!assets ? <p className="evidence-note">{error ?? "Loading measured evidence…"}</p> : <>
      <div className="report-grid">
        <article><span className="report-marker done">✓</span><h3>What worked</h3><p>Keyframe decoding, motion-aware tracking and per-clip zone alignment produced 26 reviewable events. On the team's labels, Score A was 0.2846. The measured harness runs stayed below 3× duration.</p></article>
        <article><span className="report-marker pending">!</span><h3>What did not</h3><p>Wrong-way is gated off at the sample rate; far-kerb pedestrians and a moped rider expose detector limits; the truck queue is an ambiguous missed congestion label. Part B has no accident validation. Team-authored labels are circular, and laptop CPU timing varies.</p></article>
        <article><span className="report-marker pending">→</span><h3>What comes next</h3><p>Test on an independent labelled set and a T4, add small-object and rider examples, improve signal-phase reasoning for queues, and calibrate accident risk on actual near-collision footage.</p></article>
      </div>
      <details className="report-decisions"><summary>Engineering decisions from the camera and codec measurements</summary><div className="decision-grid">{assets.format.decisions.map((item, index) => <article key={item.finding}><span>{String(index + 1).padStart(2, "0")}</span><h3>{item.finding}</h3><p>{item.decision}</p></article>)}</div></details>
      <p className="evidence-note">{assets.team.attribution}</p>
      <div className="resource-row" id="links"><a href={repo} target="_blank" rel="noreferrer">Repository ↗</a><a href={tag} target="_blank" rel="noreferrer">Verified tag ↗</a><a href={`${repo}/blob/v1.0-final2/weights/yolo11s.pt`} target="_blank" rel="noreferrer">Weights ↗</a><a href={`${repo}/blob/v1.0-final2/predictions_samples.json`} target="_blank" rel="noreferrer">Sample predictions ↗</a></div>
    </>}
  </section>;
}
