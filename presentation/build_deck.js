/* GridSentinal final deck — Group6 Literature Review visual language.
   Warm paper, navy ink, steel-blue accents, hairline rules, Georgia titles. */
const pptxgen = require("pptxgenjs");

const W = 13.33, H = 7.5, M = 0.55;
const BG = "F7F5F0";        // warm paper
const INK = "26313C";       // navy charcoal
const STEEL = "3D5A73";     // steel blue (primary accent)
const MUTED = "5B6B78";     // slate
const FAINT = "8B98A3";
const LINE = "D9D5CC";      // hairline on paper
const GREEN = "2F6B5E";     // semantic positive
const AMBER = "9A6A1F";
const RED = "8C3B31";
const TINT = "EFEBE2";      // pale panel tint
const SANS = "Arial";
const SERIF = "Georgia";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "The Unsupervised Suspects";
pres.title = "GridSentinal - Yuva Yodha Energy Tech 2026";

let pageNo = 0;
function slide(kicker, title, subtitle) {
  pageNo += 1;
  const s = pres.addSlide();
  s.background = { color: BG };
  // header zone
  s.addText(kicker, { x: M, y: 0.42, w: 9, h: 0.3, fontFace: SANS, fontSize: 10.5,
    color: STEEL, charSpacing: 3, bold: true, margin: 0 });
  s.addText(title, { x: M, y: 0.72, w: W - 2 * M, h: 0.62, fontFace: SERIF, fontSize: 27,
    color: INK, bold: true, margin: 0 });
  if (subtitle) s.addText(subtitle, { x: M, y: 1.38, w: W - 2 * M, h: 0.3, fontFace: SANS,
    fontSize: 11.5, color: MUTED, margin: 0 });
  // hairline under header
  s.addShape(pres.shapes.LINE, { x: M, y: 1.82, w: W - 2 * M, h: 0, line: { color: LINE, width: 0.75 } });
  // footer
  s.addShape(pres.shapes.LINE, { x: M, y: 7.06, w: W - 2 * M, h: 0, line: { color: LINE, width: 0.75 } });
  s.addText("GRIDSENTINAL  |  Adaptive, Self-Verifying Energy Intelligence", {
    x: M, y: 7.12, w: 6.4, h: 0.26, fontFace: SANS, fontSize: 8, color: FAINT, margin: 0 });
  s.addText("Yuva Yodha Energy Tech 2026  |  Schneider Electric", {
    x: 5.2, y: 7.12, w: 5.4, h: 0.26, fontFace: SANS, fontSize: 8, color: FAINT, align: "right", margin: 0 });
  s.addText(String(pageNo).padStart(2, "0") + " / 11", {
    x: W - M - 1.0, y: 7.12, w: 1.0, h: 0.26, fontFace: SANS, fontSize: 8, color: FAINT, align: "right", margin: 0 });
  return s;
}
const bu = () => ({ code: "2013", indent: 10 });

/* ============================== 01 PROBLEM ============================== */
(() => {
  const s = slide("01 / THE CONTEXT & PROBLEM", "THE GRID IS BECOMING\nLESS PREDICTABLE.",
    "Renewables fluctuate. Demand moves. Flexibility is uncertain. Every intervention has consequences.");

  // flow: sources -> grid state -> uncertainty -> decision
  const y0 = 2.5, boxH = 0.72, srcW = 2.05, mainW = 1.85;
  const srcs = ["RENEWABLES", "DEMAND", "DER ASSETS"];
  const gy = 3.5; // centre of the chain row
  srcs.forEach((t, i) => {
    const y = y0 + i * (boxH + 0.24);
    s.addShape(pres.shapes.RECTANGLE, { x: M, y, w: srcW, h: boxH, fill: { color: BG }, line: { color: STEEL, width: 1 } });
    s.addText(t, { x: M, y, w: srcW, h: boxH, fontFace: SANS, fontSize: 10.5, color: INK, align: "center", valign: "middle", charSpacing: 1.5, margin: 0 });
    // clean converging connector to the chain
    const sy = y + boxH / 2;
    const ex = M + srcW + 0.55, ey = gy;
    s.addShape(pres.shapes.LINE, { x: M + srcW, y: Math.min(sy, ey), w: ex - (M + srcW), h: Math.abs(ey - sy), line: { color: FAINT, width: 1 }, flipV: sy > ey });
  });
  const chain = [
    { t: "GRID STATE", d: "5,196 nodes\n1,871 loads" },
    { t: "UNCERTAINTY", d: "intervals, not\nsingle numbers" },
    { t: "DECISION", d: "act or hold" },
  ];
  let cx = M + srcW + 0.55;
  chain.forEach((c, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: gy - 0.95, w: mainW, h: 1.9, fill: { color: TINT }, line: { color: LINE, width: 0.75 } });
    s.addText(c.t, { x: cx, y: gy - 0.78, w: mainW, h: 0.3, fontFace: SANS, fontSize: 11, bold: true, color: INK, align: "center", margin: 0 });
    s.addText(c.d, { x: cx, y: gy - 0.4, w: mainW, h: 0.9, fontFace: SANS, fontSize: 9.5, color: MUTED, align: "center", margin: 0 });
    if (i < chain.length - 1) {
      s.addText("\u2192", { x: cx + mainW, y: gy - 0.2, w: 0.42, h: 0.4, fontFace: SANS, fontSize: 16, color: STEEL, align: "center", margin: 0 });
    }
    cx += mainW + 0.42;
  });

  // data foundation row
  s.addShape(pres.shapes.LINE, { x: M, y: 5.35, w: W - 2 * M, h: 0, line: { color: LINE, width: 0.75 } });
  s.addText("MODELED ON REAL DATA  ·  SMART-DS v1.0", { x: M, y: 5.5, w: 5, h: 0.26, fontFace: SANS, fontSize: 9.5, color: STEEL, charSpacing: 2, bold: true, margin: 0 });
  const facts = [["5,196", "nodes"], ["1,871", "customer loads"], ["1,216", "PV arrays"], ["93", "batteries"], ["35,040", "15-min intervals"]];
  facts.forEach(([v, l], i) => {
    const x = M + i * 2.45;
    s.addText(v, { x, y: 5.82, w: 2.2, h: 0.42, fontFace: SANS, fontSize: 21, bold: true, color: INK, margin: 0 });
    s.addText(l, { x, y: 6.26, w: 2.2, h: 0.26, fontFace: SANS, fontSize: 9.5, color: MUTED, margin: 0 });
  });
  s.addText("Source: SMART-DS v1.0 feeder p1uhs0_1247, AUS/P1U 2018; verified ingestion, 359 files SHA-256 checked.", {
    x: M, y: 6.66, w: W - 2 * M, h: 0.24, fontFace: SANS, fontSize: 9, color: FAINT, margin: 0 });
})();

/* ============================== 02 WHY HARD ============================== */
(() => {
  const s = slide("02 / THE CRITICAL INDUSTRY GAP", "PREDICTION IS ONLY\nTHE FIRST PROBLEM.",
    "Why naive automation fails: four interacting challenges, measured on a real feeder.");

  const quads = [
    ["01", "VARIABLE DEMAND", "Household load on a 15-minute grid is smooth at 15 min but error grows with horizon: MAE 0.40 kW at 15 min vs 2.43 kW at 24 h for persistence."],
    ["02", "RENEWABLE INTERMITTENCY", "Solar ramps concentrate error. Without a weather forecast, day-ahead PV error is 74.3 kW against 8.57 kW with weather at the origin."],
    ["03", "UNCERTAIN FLEXIBILITY", "0 of 8 physical flexibility dimensions are supported by the data. A statistical proxy exists, but it is not dispatchable capacity."],
    ["04", "UNCHECKED DECISION RISK", "A point forecast hides its own reliability. The default constant-width interval fails 11 of 12 coverage cells on this feeder."],
  ];
  const qw = (W - 2 * M - 0.5) / 2, qh = 1.78;
  quads.forEach(([n, t, d], i) => {
    const x = M + (i % 2) * (qw + 0.5), y = 2.1 + Math.floor(i / 2) * (qh + 0.28);
    s.addText(n, { x, y, w: 0.7, h: 0.5, fontFace: SERIF, fontSize: 24, color: STEEL, margin: 0 });
    s.addText(t, { x: x + 0.72, y: y + 0.06, w: qw - 0.72, h: 0.3, fontFace: SANS, fontSize: 12, bold: true, color: INK, charSpacing: 1, margin: 0 });
    s.addText(d, { x: x + 0.72, y: y + 0.42, w: qw - 0.85, h: qh - 0.5, fontFace: SANS, fontSize: 10, color: MUTED, margin: 0 });
    if (i < 2) s.addShape(pres.shapes.LINE, { x, y: y + qh + 0.14, w: qw, h: 0, line: { color: LINE, width: 0.75 } });
  });

  // formula strip
  const fy = 6.18;
  const terms = ["FORECAST ERROR", "HIDDEN UNCERTAINTY", "UNKNOWN CAPABILITY", "PHYSICAL CONSEQUENCES"];
  let fx = M;
  const tw = 2.14;
  terms.forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: tw, h: 0.5, fill: { color: TINT }, line: { color: LINE, width: 0.75 } });
    s.addText(t, { x: fx, y: fy, w: tw, h: 0.5, fontFace: SANS, fontSize: 8.6, bold: true, color: INK, align: "center", valign: "middle", margin: 0 });
    if (i < 3) s.addText("+", { x: fx + tw, y: fy, w: 0.3, h: 0.5, fontFace: SANS, fontSize: 14, color: STEEL, align: "center", valign: "middle", margin: 0 });
    fx += tw + 0.3;
  });
  s.addText("=", { x: fx, y: fy, w: 0.3, h: 0.5, fontFace: SANS, fontSize: 14, color: STEEL, align: "center", valign: "middle", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: fx + 0.34, y: fy, w: 2.6, h: 0.5, fill: { color: INK } });
  s.addText("NAIVE AUTOMATION IS UNSAFE", { x: fx + 0.34, y: fy, w: 2.6, h: 0.5, fontFace: SANS, fontSize: 9.5, bold: true, color: "FFFFFF", align: "center", valign: "middle", margin: 0 });
})();

/* ============================== 03 SOLUTION ============================== */
(() => {
  const s = slide("03 / PROPOSED SOLUTION", "MEET GRIDSENTINAL.",
    "An adaptive, self-verifying energy intelligence system for renewable-integrated distributed networks.");

  // left: philosophy quote + pipeline
  s.addShape(pres.shapes.LINE, { x: M, y: 2.3, w: 0.05, h: 1.15, line: { color: STEEL, width: 2.5 } });
  s.addText("\u201CDon't just predict the grid.\nVerify what happens next.\u201D", {
    x: M + 0.25, y: 2.28, w: 5.2, h: 1.2, fontFace: SERIF, fontSize: 19, italic: true, color: INK, margin: 0 });
  s.addText("Uncertainty, capability limits and decision verification are first-class parts of the pipeline, not reporting dashboards.", {
    x: M + 0.25, y: 3.55, w: 4.9, h: 0.75, fontFace: SANS, fontSize: 10.5, color: MUTED, margin: 0 });

  const steps = ["FORECAST", "UNCERTAINTY", "FLEXIBILITY", "DECISION", "VERIFY"];
  let sx = M + 0.25;
  steps.forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: sx, y: 4.62, w: 1.06, h: 0.42, fill: { color: i < 2 ? STEEL : TINT }, line: { color: i < 2 ? STEEL : LINE, width: 0.75 } });
    s.addText(t, { x: sx, y: 4.62, w: 1.06, h: 0.42, fontFace: SANS, fontSize: 7.6, bold: true, color: i < 2 ? "FFFFFF" : INK, align: "center", valign: "middle", margin: 0 });
    if (i < steps.length - 1) s.addText("\u2193", { x: sx + 1.06, y: 4.62, w: 0.24, h: 0.42, fontFace: SANS, fontSize: 11, color: STEEL, align: "center", valign: "middle", margin: 0 });
    sx += 1.3;
  });
  s.addText("Shaded: operational today. Outlined: designed, next on the roadmap.", {
    x: M + 0.25, y: 5.18, w: 5.2, h: 0.26, fontFace: SANS, fontSize: 9, color: FAINT, margin: 0 });

  // right: framed console screenshot
  const ix = 6.62, iy = 2.14, iw = 6.16, ih = 3.46; // 1600x655 crop ratio 2.44 -> h 2.52; widen
  const cw = 6.16, ch = cw * (559 / 1600);
  s.addShape(pres.shapes.RECTANGLE, { x: ix - 0.06, y: iy - 0.06, w: cw + 0.12, h: ch + 0.12, fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 }, shadow: { type: "outer", color: "26313C", blur: 8, offset: 2, angle: 90, opacity: 0.18 } });
  s.addImage({ path: "web/shots/deck/console-crop.png", x: ix, y: iy, w: cw, h: ch });
  s.addText("GridSentinal console: 48-hour demand forecast with calibrated 90% interval, recorded model output on SMART-DS telemetry.", {
    x: ix, y: iy + ch + 0.12, w: cw, h: 0.4, fontFace: SANS, fontSize: 9, color: MUTED, margin: 0 });
})();

/* ============================== 04 HOW IT WORKS ============================== */
(() => {
  const s = slide("04 / SYSTEM ARCHITECTURE", "FROM TELEMETRY\nTO TRUSTED DECISION.",
    "A layered, evidence-based pipeline with explicit operational status for every layer.");

  const nodes = [
    ["DATA", "OPERATIONAL"], ["DATA QUALITY", "OPERATIONAL"], ["FORECASTING", "OPERATIONAL"],
    ["EXPERTS", "OPERATIONAL"], ["UNCERTAINTY", "OPERATIONAL"], ["FLEXIBILITY", "STATISTICAL PROXY"],
    ["DECISION", "DESIGNED"], ["ASSURANCE", "DESIGNED"],
  ];
  const nw = 1.38, gap = 0.17, y0 = 2.5, nh = 1.5;
  nodes.forEach(([t, st], i) => {
    const x = M + i * (nw + gap);
    const live = st === "OPERATIONAL";
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w: nw, h: nh, fill: { color: live ? TINT : BG }, line: { color: live ? STEEL : LINE, width: live ? 1 : 0.75 } });
    s.addText(t, { x: x + 0.06, y: y0 + 0.14, w: nw - 0.12, h: 0.62, fontFace: SANS, fontSize: 9.5, bold: true, color: INK, margin: 0 });
    s.addText(st, { x: x + 0.06, y: y0 + nh - 0.44, w: nw - 0.12, h: 0.34, fontFace: SANS, fontSize: 7.2,
      color: st === "OPERATIONAL" ? GREEN : st === "STATISTICAL PROXY" ? AMBER : STEEL, charSpacing: 0.5, margin: 0 });
    if (i < nodes.length - 1) s.addText("\u2192", { x: x + nw - 0.03, y: y0 + 0.55, w: gap + 0.06, h: 0.4, fontFace: SANS, fontSize: 12, color: STEEL, align: "center", margin: 0 });
  });

  // annotations under the two halves
  s.addText("WHAT RUNS TODAY", { x: M, y: 4.35, w: 5, h: 0.26, fontFace: SANS, fontSize: 9.5, bold: true, color: GREEN, charSpacing: 2, margin: 0 });
  s.addText("Data ingestion and reality validation, multi-expert forecasting, and calibrated uncertainty are operational end to end on 179,520 sealed test rows.", {
    x: M, y: 4.62, w: 5.6, h: 0.75, fontFace: SANS, fontSize: 10, color: MUTED, margin: 0 });
  s.addText("WHAT COMES NEXT", { x: 7.1, y: 4.35, w: 5, h: 0.26, fontFace: SANS, fontSize: 9.5, bold: true, color: STEEL, charSpacing: 2, margin: 0 });
  s.addText("Flexibility today is a statistical proxy (a data blocker is recorded). The decision, red-team, twin and assurance layers are designed and specified, not implemented.", {
    x: 7.1, y: 4.62, w: 5.55, h: 0.75, fontFace: SANS, fontSize: 10, color: MUTED, margin: 0 });

  s.addShape(pres.shapes.LINE, { x: M, y: 5.62, w: W - 2 * M, h: 0, line: { color: LINE, width: 0.75 } });
  s.addText([
    { text: "Design contract:  ", options: { bold: true, color: INK } },
    { text: "no layer is marked operational that has not been run end to end against the sealed test split. Roadmap components are labelled as such throughout this deck.", options: { color: MUTED } },
  ], { x: M, y: 5.78, w: W - 2 * M, h: 0.55, fontFace: SANS, fontSize: 10.5, margin: 0 });
})();

/* ============================== 05 UNCERTAINTY + FLEXIBILITY ============================== */
(() => {
  const s = slide("05 / PROBABILISTIC INTELLIGENCE", "KNOW WHAT YOU DON'T KNOW.",
    "A point forecast fails silently. A calibrated interval says which rows to trust, and which to suppress.");

  // left: native line chart, real 48h slice (every 4th point of 192)
  const ev = require("../web/src/data/evidence.json");
  const sl = ev.forecast_slice;
  const idx = sl.point_kw.filter((_, i) => i % 4 === 0);
  const labels = idx.map((_, i) => (i % 8 === 0 ? sl.timestamps[i * 4].slice(11, 16) : ""));
  s.addChart(pres.charts.LINE, [
    { name: "Upper 90%", labels, values: sl.upper_kw.filter((_, i) => i % 4 === 0) },
    { name: "Point forecast", labels, values: idx },
    { name: "Lower 90%", labels, values: sl.lower_kw.filter((_, i) => i % 4 === 0) },
  ], {
    x: M, y: 2.14, w: 6.3, h: 3.0,
    chartColors: ["C9D4DC", STEEL, "C9D4DC"],
    lineDataSymbol: "none", lineSize: 1.5, lineSmooth: true,
    catAxisLabelColor: FAINT, valAxisLabelColor: FAINT, catAxisLabelFontSize: 8, valAxisLabelFontSize: 8,
    valGridLine: { color: "E5E1D8", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: false, valAxisTitle: "kW", showValAxisTitle: true, valAxisTitleFontSize: 9,
    chartArea: { fill: { color: BG } }, plotArea: { fill: { color: BG } },
  });
  s.addText("48-hour window, 1-hour-ahead horizon, asset load_p1ulv14763. Recorded output, sealed test split.", {
    x: M, y: 5.2, w: 6.3, h: 0.26, fontFace: SANS, fontSize: 9, color: FAINT, margin: 0 });

  // coverage callouts
  const cov = [["90.2%", "coverage at h=1"], ["89.7%", "coverage at h=4"], ["+0.67", "width-error rank corr."]];
  cov.forEach(([v, l], i) => {
    const x = M + i * 2.14;
    s.addText(v, { x, y: 5.56, w: 2.0, h: 0.4, fontFace: SANS, fontSize: 19, bold: true, color: GREEN, margin: 0 });
    s.addText(l, { x, y: 5.97, w: 2.0, h: 0.24, fontFace: SANS, fontSize: 9, color: MUTED, margin: 0 });
  });

  // right: flexibility truth panel
  const rx = 7.25;
  s.addText("FLEXIBILITY WITHOUT FICTION", { x: rx, y: 2.14, w: 5.4, h: 0.28, fontFace: SANS, fontSize: 10.5, bold: true, color: STEEL, charSpacing: 2, margin: 0 });
  const rows = [
    ["PHYSICAL", "UNKNOWN", "0 of 8 flexibility dimensions supported by the dataset (G-01).", RED],
    ["STATISTICAL", "AVAILABLE", "Behavioural envelope from 1,871 consumption series; rank correlation with error confirms signal.", AMBER],
    ["ASSUMED", "SCENARIO ONLY", "Dispatch scenarios stay clearly labelled as assumptions.", MUTED],
  ];
  rows.forEach(([k, v, d, c], i) => {
    const y = 2.56 + i * 0.98;
    s.addText(k, { x: rx, y, w: 1.7, h: 0.3, fontFace: SANS, fontSize: 10.5, bold: true, color: INK, margin: 0 });
    s.addText(v, { x: rx + 1.75, y, w: 1.9, h: 0.3, fontFace: SANS, fontSize: 10.5, bold: true, color: c, margin: 0 });
    s.addText(d, { x: rx, y: y + 0.32, w: 5.3, h: 0.5, fontFace: SANS, fontSize: 9.5, color: MUTED, margin: 0 });
    if (i < 2) s.addShape(pres.shapes.LINE, { x: rx, y: y + 0.86, w: 5.35, h: 0, line: { color: LINE, width: 0.75 } });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: rx, y: 5.62, w: 5.4, h: 0.72, fill: { color: TINT } });
  s.addText("Historical consumption behaviour is not automatically dispatchable capacity.", {
    x: rx + 0.15, y: 5.62, w: 5.1, h: 0.72, fontFace: SERIF, fontSize: 12, italic: true, color: INK, valign: "middle", margin: 0 });
})();

/* ============================== 06 DECISION ASSURANCE ============================== */
(() => {
  const s = slide("06 / DECISION ASSURANCE ENGINE", "BEFORE IT ACTS,\nIT TRIES TO BREAK THE DECISION.",
    "Instead of asking only what the system should do, GridSentinal asks whether the action survives hostile evaluation.");

  const steps = [
    ["01", "PROPOSE", "Optimizer drafts a candidate action from forecasts, uncertainty, flexibility and constraints."],
    ["02", "RED TEAM", "Adversarial scenarios attack failure modes: when is it unsafe, ineffective or suboptimal?"],
    ["03", "SIMULATE", "An independent digital twin replays the action against physics and compares outcomes."],
    ["04", "ASSURE", "The assurance gate approves, rejects, or re-optimizes, with human-in-the-loop fallback."],
    ["05", "EXECUTE", "Only after verification. Outcomes return to the system as performance data."],
  ];
  const sw = 2.28, gap = 0.19, y0 = 2.3;
  steps.forEach(([n, t, d], i) => {
    const x = M + i * (sw + gap);
    s.addText(n, { x, y: y0, w: 0.6, h: 0.42, fontFace: SERIF, fontSize: 20, color: STEEL, margin: 0 });
    s.addText(t, { x, y: y0 + 0.44, w: sw - 0.1, h: 0.3, fontFace: SANS, fontSize: 11.5, bold: true, color: INK, charSpacing: 1, margin: 0 });
    s.addText(d, { x, y: y0 + 0.78, w: sw - 0.14, h: 1.25, fontFace: SANS, fontSize: 9.5, color: MUTED, margin: 0 });
    if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + sw + 0.02, y: y0 + 0.16, w: gap - 0.04, h: 0, line: { color: LINE, width: 1 } });
  });

  // status strip
  s.addShape(pres.shapes.LINE, { x: M, y: 4.6, w: W - 2 * M, h: 0, line: { color: LINE, width: 0.75 } });
  s.addText([
    { text: "STATUS.  ", options: { bold: true, color: INK } },
    { text: "The assurance sequence is a designed architecture (roadmap), not shipped functionality. What is shipped today is the discipline it will enforce: sealed data splits, pre-registered success bars, and negative results recorded instead of engineered around.", options: { color: MUTED } },
  ], { x: M, y: 4.76, w: W - 2 * M, h: 0.7, fontFace: SANS, fontSize: 10.5, margin: 0 });

  s.addShape(pres.shapes.RECTANGLE, { x: M, y: 5.75, w: W - 2 * M, h: 0.82, fill: { color: INK } });
  s.addText("\u201CAn action is not executed because it was proposed. It is executed because it survived.\u201D", {
    x: M + 0.3, y: 5.75, w: W - 2 * M - 0.6, h: 0.82, fontFace: SERIF, fontSize: 14, italic: true, color: "FFFFFF", valign: "middle", margin: 0 });
})();

/* ============================== 07 TECHNICAL APPROACH ============================== */
(() => {
  const s = slide("07 / TECHNICAL APPROACH", "A RESEARCH PIPELINE,\nNOT A SINGLE MODEL.",
    "Five layers, each with an explicit implementation status and a measurable exit criterion.");

  const layers = [
    ["LAYER 1", "DATA", "SMART-DS v1.0, 15-minute native resolution, 359 files checksummed", "IMPLEMENTED", true],
    ["LAYER 2", "FORECASTING", "Persistence, Hist-GBM, temporal TCN; fixed weighted ensemble ships", "IMPLEMENTED", true],
    ["LAYER 3", "PROBABILISTIC INTELLIGENCE", "Split-conformal intervals, six methods calibrated, one published procedure", "IMPLEMENTED", true],
    ["LAYER 4", "FLEXIBILITY", "Capability audit, behavioural envelope, physical gaps recorded", "PARTIAL", false],
    ["LAYER 5", "DECISION", "Optimization, red team, digital twin, assurance gate", "DESIGNED", false],
  ];
  const y0 = 2.18, lh = 0.78;
  layers.forEach(([k, t, d, st, live], i) => {
    const y = y0 + i * (lh + 0.12);
    s.addShape(pres.shapes.RECTANGLE, { x: M, y, w: W - 2 * M, h: lh, fill: { color: live ? TINT : BG }, line: { color: live ? STEEL : LINE, width: live ? 1 : 0.75 } });
    s.addText(k, { x: M + 0.2, y, w: 0.95, h: lh, fontFace: SANS, fontSize: 9, bold: true, color: FAINT, valign: "middle", margin: 0 });
    s.addText(t, { x: M + 1.2, y, w: 3.3, h: lh, fontFace: SANS, fontSize: 12, bold: true, color: INK, valign: "middle", margin: 0 });
    s.addText(d, { x: M + 4.6, y, w: 5.9, h: lh, fontFace: SANS, fontSize: 10, color: MUTED, valign: "middle", margin: 0 });
    s.addText(st, { x: M + 10.7, y, w: 1.8, h: lh, fontFace: SANS, fontSize: 9, bold: true, color: live ? GREEN : STEEL, valign: "middle", align: "right", charSpacing: 1, margin: 0 });
  });

  s.addText("TECH STACK", { x: M, y: 6.6, w: 1.4, h: 0.26, fontFace: SANS, fontSize: 9.5, bold: true, color: STEEL, charSpacing: 2, margin: 0 });
  s.addText("Python 3.12  ·  PyTorch (CPU-deterministic)  ·  scikit-learn  ·  OpenDSS  ·  pytest (1,235 tests)", {
    x: M + 1.5, y: 6.6, w: 9.5, h: 0.26, fontFace: SANS, fontSize: 10, color: MUTED, margin: 0 });
})();

/* ============================== 08 RESEARCH EVIDENCE ============================== */
(() => {
  const s = slide("08 / RESEARCH & EXPERIMENTAL EVIDENCE", "WE TESTED THE IDEA.\nWE DIDN'T JUST DEMO IT.",
    "1,235 tests passing, 29 registered experiments, and negative results recorded rather than engineered around.");

  const studies = [
    ["QWEN", "Language-model specialization", "NEGATIVE", RED],
    ["ROUTER", "Learned expert routing", "NEGATIVE", RED],
    ["UNCERTAINTY", "Conformal calibration", "VERIFIED", GREEN],
    ["FLEXIBILITY", "Physical capability audit", "NO PHYSICAL SUPPORT", AMBER],
    ["ABLATION", "Feature contribution", "TARGET-DEPENDENT", AMBER],
  ];
  const tw = 2.28, gap = 0.19, y0 = 2.16;
  studies.forEach(([k, t, v, c], i) => {
    const x = M + i * (tw + gap);
    s.addShape(pres.shapes.LINE, { x, y: y0, w: tw - 0.15, h: 0, line: { color: LINE, width: 0.75 } });
    s.addText(k, { x, y: y0 + 0.1, w: tw - 0.15, h: 0.26, fontFace: SANS, fontSize: 9.5, bold: true, color: INK, charSpacing: 1.5, margin: 0 });
    s.addText(t, { x, y: y0 + 0.38, w: tw - 0.15, h: 0.45, fontFace: SANS, fontSize: 9, color: MUTED, margin: 0 });
    s.addText(v, { x, y: y0 + 0.86, w: tw - 0.15, h: 0.26, fontFace: SANS, fontSize: 8.5, bold: true, color: c, charSpacing: 0.5, margin: 0 });
  });

  // chart: 24h-ahead MAE by model (kW), sealed split
  s.addChart(pres.charts.BAR, [{
    name: "MAE at 24 h (kW)",
    labels: ["Persistence", "Hist-GBM", "Temporal TCN", "Learned router", "Fixed ensemble", "Oracle bound"],
    values: [2.4343, 1.5737, 1.6510, 1.5600, 1.5399, 0.9942],
  }], {
    x: M, y: 3.45, w: 6.4, h: 3.0, barDir: "bar",
    chartColors: ["C9D4DC", "C9D4DC", "C9D4DC", "C9D4DC", GREEN, "9FB4C4"],
    varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK, dataLabelFontSize: 9, dataLabelFormatCode: "0.00",
    catAxisLabelColor: INK, catAxisLabelFontSize: 9.5, valAxisHidden: true,
    valGridLine: { style: "none" }, catGridLine: { style: "none" },
    showLegend: false, chartArea: { fill: { color: BG } }, valAxisMaxVal: 2.8,
  });
  s.addText("Demand forecast error at 24 h ahead (kW MAE, sealed test split, 179,520 rows). The fixed weighted ensemble ships; the oracle bounds what perfect routing would reach.", {
    x: M, y: 6.5, w: 6.4, h: 0.45, fontFace: SANS, fontSize: 9, color: FAINT, margin: 0 });

  // right callouts
  const rx = 7.35;
  s.addText("WHAT THE NUMBERS SAY", { x: rx, y: 3.45, w: 5, h: 0.26, fontFace: SANS, fontSize: 10, bold: true, color: STEEL, charSpacing: 2, margin: 0 });
  const call = [
    ["+0.66 to +0.71", "rank correlation between interval width and realized error at every horizon"],
    ["4.2x to 4.8x", "mean error carried by the widest tenth of prediction intervals"],
    ["11 of 12", "coverage cells failed by the default constant-width interval (recorded, not hidden)"],
  ];
  call.forEach(([v, d], i) => {
    const y = 3.82 + i * 0.92;
    s.addText(v, { x: rx, y, w: 5.2, h: 0.36, fontFace: SANS, fontSize: 17, bold: true, color: INK, margin: 0 });
    s.addText(d, { x: rx, y: y + 0.38, w: 5.25, h: 0.45, fontFace: SANS, fontSize: 9.5, color: MUTED, margin: 0 });
  });
})();

/* ============================== 09 INNOVATION + ALIGNMENT ============================== */
(() => {
  const s = slide("09 / INNOVATION & CHALLENGE ALIGNMENT", "WHAT ACTUALLY MAKES\nGRIDSENTINAL DIFFERENT?",
    "A direct causal mapping from the Grid Reliability challenge needs to system capabilities.");

  s.addText("CHALLENGE NEED", { x: M + 0.1, y: 2.05, w: 3.4, h: 0.26, fontFace: SANS, fontSize: 9.5, bold: true, color: FAINT, charSpacing: 2, margin: 0 });
  s.addText("GRIDSENTINAL RESPONSE", { x: 4.75, y: 2.05, w: 3.9, h: 0.26, fontFace: SANS, fontSize: 9.5, bold: true, color: FAINT, charSpacing: 2, margin: 0 });
  s.addText("VALUE", { x: 9.35, y: 2.05, w: 3.3, h: 0.26, fontFace: SANS, fontSize: 9.5, bold: true, color: FAINT, charSpacing: 2, margin: 0 });

  const rows = [
    ["RENEWABLE INTERMITTENCY", "15-min multi-expert nowcasting plus a calibrated 24 h horizon with conformal error envelopes.", "Ramp foresight for DISCOMs; no blind generation shortfalls."],
    ["DER & STORAGE COORDINATION", "Flexibility capability audit and behavioural envelopes; refuses unverified dispatch.", "No over-commitment penalties; only physically supported assets move."],
    ["TECHNICAL LOSSES & PHYSICS", "Formal energy contract with loss modeling; OpenDSS twin validates power flow.", "Dispatch actions close physically; thermal stress prevented."],
    ["AUTONOMOUS GRID SAFETY", "Red-team stress testing plus a decision-assurance gate with human fallback.", "Zero unchecked automated decisions; verifiable operational safety."],
  ];
  rows.forEach(([a, b, c], i) => {
    const y = 2.42 + i * 0.98;
    s.addText(a, { x: M, y, w: 3.5, h: 0.6, fontFace: SANS, fontSize: 10.5, bold: true, color: INK, margin: 0 });
    s.addText("\u2192", { x: 4.05, y, w: 0.4, h: 0.6, fontFace: SANS, fontSize: 13, color: STEEL, align: "center", margin: 0 });
    s.addText(b, { x: 4.55, y, w: 3.9, h: 0.85, fontFace: SANS, fontSize: 10, color: MUTED, margin: 0 });
    s.addText("\u2192", { x: 8.6, y, w: 0.4, h: 0.6, fontFace: SANS, fontSize: 13, color: STEEL, align: "center", margin: 0 });
    s.addText(c, { x: 9.1, y, w: 3.6, h: 0.85, fontFace: SANS, fontSize: 10, color: MUTED, margin: 0 });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: M, y: y + 0.88, w: W - 2 * M, h: 0, line: { color: LINE, width: 0.75 } });
  });

  s.addShape(pres.shapes.RECTANGLE, { x: M, y: 6.42, w: W - 2 * M, h: 0.52, fill: { color: TINT } });
  s.addText([
    { text: "SYSTEM-LEVEL INNOVATION.  ", options: { bold: true, color: INK } },
    { text: "GridSentinal treats uncertainty, capability limits and verification as first-class parts of energy intelligence.", options: { color: MUTED } },
  ], { x: M + 0.2, y: 6.42, w: W - 2 * M - 0.4, h: 0.52, fontFace: SANS, fontSize: 10.5, valign: "middle", margin: 0 });
})();

/* ============================== 10 IMPACT + ROADMAP ============================== */
(() => {
  const s = slide("10 / VALUE CREATION & ROADMAP", "FROM LAB EVIDENCE\nTO INDUSTRIAL DEPLOYMENT.",
    "A clear line between the completed research foundation and industrial scaling.");

  s.addText("EXPECTED IMPACT", { x: M, y: 2.1, w: 5, h: 0.26, fontFace: SANS, fontSize: 10, bold: true, color: STEEL, charSpacing: 2, margin: 0 });
  const impact = [
    ["GRID RELIABILITY", "Pre-emptive mitigation of solar ramps and evening peaks through calibrated forecast bounds."],
    ["DISCOM OPERATIONS", "Visibility into demand behaviour; statistical proxies never mistaken for dispatchable capacity."],
    ["OPERATIONAL SAFETY", "Verified decisions instead of blind automation; every consequential action confirmed."],
    ["ENERGY-AWARE MLOPS", "Model performance judged by operational consequence, with deployment gates."],
  ];
  impact.forEach(([t, d], i) => {
    const y = 2.52 + i * 1.02;
    s.addText(t, { x: M, y, w: 5.6, h: 0.28, fontFace: SANS, fontSize: 11, bold: true, color: INK, charSpacing: 0.5, margin: 0 });
    s.addText(d, { x: M, y: y + 0.3, w: 5.5, h: 0.6, fontFace: SANS, fontSize: 10, color: MUTED, margin: 0 });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: M, y: y + 0.9, w: 5.6, h: 0, line: { color: LINE, width: 0.75 } });
  });

  s.addShape(pres.shapes.LINE, { x: 6.9, y: 2.1, w: 0, h: 4.7, line: { color: LINE, width: 0.75 } });

  s.addText("ROADMAP", { x: 7.25, y: 2.1, w: 5, h: 0.26, fontFace: SANS, fontSize: 10, bold: true, color: STEEL, charSpacing: 2, margin: 0 });
  const road = [
    ["01", "PROVEN FOUNDATION", "Data validation, forecasting ensemble, calibrated uncertainty, flexibility audit, feature ablation.", "COMPLETE", GREEN],
    ["02", "DECISION & VERIFICATION", "Optimization, digital twin, red team, assurance gate; ADMS / DERMS integration paths.", "NEXT", STEEL],
    ["03", "INDUSTRIAL SCALING", "Hardware-in-the-loop testing, MLOps deployment gates, multi-feeder operation.", "FUTURE", FAINT],
  ];
  road.forEach(([n, t, d, st, c], i) => {
    const y = 2.52 + i * 1.42;
    s.addText(n, { x: 7.25, y, w: 0.62, h: 0.5, fontFace: SERIF, fontSize: 22, color: c, margin: 0 });
    s.addText(t, { x: 7.95, y: y + 0.03, w: 4.6, h: 0.28, fontFace: SANS, fontSize: 11.5, bold: true, color: INK, margin: 0 });
    s.addText(st, { x: 7.95, y: y + 0.32, w: 4.6, h: 0.24, fontFace: SANS, fontSize: 8.5, bold: true, color: c, charSpacing: 1.5, margin: 0 });
    s.addText(d, { x: 7.95, y: y + 0.58, w: 4.7, h: 0.7, fontFace: SANS, fontSize: 9.5, color: MUTED, margin: 0 });
  });
})();

/* ============================== 11 TEAM + CLOSING ============================== */
(() => {
  const s = slide("11 / TEAM & CONCLUSION", "BUILDING THE GRID'S\nINTELLIGENCE LAYER.",
    "The Unsupervised Suspects · Yuva Yodha Energy Tech Hackathon 2026, Schneider Electric.");

  const team = [
    ["Pratham Kapoor", "A028"], ["Saransh Naruka", "A071"], ["Ritesh Pandey", "A086"],
    ["Moinuddin Shaikh", "A093"], ["Javin Sharma", "B091"],
  ];
  const tw = 2.3;
  team.forEach(([n, id], i) => {
    const x = M + i * (tw + 0.05);
    s.addText(n, { x, y: 2.3, w: tw, h: 0.3, fontFace: SANS, fontSize: 12.5, bold: true, color: INK, margin: 0 });
    s.addText(id, { x, y: 2.62, w: tw, h: 0.26, fontFace: SANS, fontSize: 10, color: FAINT, margin: 0 });
    if (i < team.length - 1) s.addShape(pres.shapes.LINE, { x: x + tw - 0.05, y: 2.3, w: 0, h: 0.6, line: { color: LINE, width: 0.75 } });
  });

  // framed hero screenshot
  const iw = 5.3, ih = iw * (810 / 1600);
  s.addShape(pres.shapes.RECTANGLE, { x: W - M - iw - 0.06, y: 3.15 - 0.06, w: iw + 0.12, h: ih + 0.12, fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 }, shadow: { type: "outer", color: "26313C", blur: 8, offset: 2, angle: 90, opacity: 0.18 } });
  s.addImage({ path: "web/shots/deck/hero-crop.png", x: W - M - iw, y: 3.15, w: iw, h: ih });
  s.addText("The GridSentinal public experience and console: github.com/PrathamKapoor/GridSentinal", {
    x: W - M - iw, y: 3.15 + ih + 0.12, w: iw, h: 0.3, fontFace: SANS, fontSize: 9, color: MUTED, margin: 0 });

  s.addText("GRID SENTINAL", { x: M, y: 3.3, w: 6, h: 0.3, fontFace: SANS, fontSize: 12, bold: true, color: STEEL, charSpacing: 3, margin: 0 });
  s.addText("DON'T JUST AUTOMATE THE GRID.\nMAKE IT THINK BEFORE IT ACTS.", {
    x: M, y: 3.7, w: 6.2, h: 1.3, fontFace: SERIF, fontSize: 21, bold: true, color: INK, margin: 0 });
  s.addText("github.com/PrathamKapoor/GridSentinal", {
    x: M, y: 5.15, w: 6, h: 0.3, fontFace: SANS, fontSize: 11.5, color: STEEL, margin: 0 });
  s.addText("Proprietary. All rights reserved. Research system; no production deployment is claimed.", {
    x: M, y: 6.55, w: W - 2 * M, h: 0.26, fontFace: SANS, fontSize: 9, color: FAINT, margin: 0 });
})();

pres.writeFile({ fileName: "presentation/GridSentinal_Hackathon_2026_Final.pptx" })
  .then(() => console.log("deck written"));
