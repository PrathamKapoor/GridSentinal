/**
 * Landing-page data.
 *
 * Every value here is copied from a measured artifact in this repository —
 * `web/src/data/evidence.json`, `artifacts/phase9/flexibility-main/`, or the
 * `energy-intelligence console status` ledger. Nothing is estimated, rounded
 * in the product's favour, or presented as a live feed.
 *
 * Status vocabulary (see README "Research status"):
 *   live      — implemented and run end to end against the sealed test split
 *   designed  — specified in the repo, not implemented
 *   record    — a recorded measurement from an earlier phase
 *   scope     — explicitly out of scope for this dataset
 */

export type CapabilityStatus = "live" | "designed" | "record" | "scope";

/** Section 03 — heterogeneous forecasting experts. Phase 7 registry, kW MAE. */
export interface Expert {
  key: string;
  name: string;
  family: string;
  blurb: string;
  /** MAE in kW at h=1 / h=4 / h=96 on the sealed test split. */
  h: [number, number, number];
  status: CapabilityStatus;
}

export const EXPERTS: Expert[] = [
  {
    key: "persistence",
    name: "Persistence",
    family: "naive baseline",
    blurb:
      "Tomorrow is today. Fast, untrainable, and genuinely hard to beat at 15 minutes — every model in the study has to clear it.",
    h: [0.4043, 0.8897, 2.4343],
    status: "live",
  },
  {
    key: "gbm",
    name: "Gradient boosting",
    family: "classical tabular",
    blurb:
      "Histogram GBM over calendar and lag features. The strongest single expert at 24 hours, and the trunk the language-model study lost to.",
    h: [0.4242, 0.8576, 1.5737],
    status: "live",
  },
  {
    key: "tcn",
    name: "Temporal TCN",
    family: "neural sequence",
    blurb:
      "Dilated causal convolution over the native 15-minute series. Wins the middle horizon, loses the long one, and says so.",
    h: [0.4178, 0.8307, 1.651],
    status: "live",
  },
];

/** The shipped combination, for contrast with the three experts. */
export const ENSEMBLE = {
  name: "Fixed weighted ensemble",
  h: [0.3983, 0.7917, 1.5399] as [number, number, number],
  note: "ships — beat the learned router at 3 of 3 horizons",
};

export const EXPERT_FACTS = {
  disagreementMin: 58.8,
  disagreementMax: 75.2,
  horizons: ["15 min", "1 h", "24 h"],
  routerBeatsEnsemble: 0,
};

/** Section 06 — the three flexibility bases. Phase 9 capability audit. */
export interface FlexLayer {
  key: "physical" | "statistical" | "assumed";
  label: string;
  value: string;
  tone: "unknown" | "available" | "scenario";
  body: string;
  measures: [string, string][];
}

export const FLEX_LAYERS: FlexLayer[] = [
  {
    key: "physical",
    label: "Physical",
    value: "Unknown",
    tone: "unknown",
    body: "Whether an asset can actually be commanded. Audited across eight dimensions against the dataset's own capability metadata; none carried a limit and a control interface together.",
    measures: [
      ["Dimensions audited", "8"],
      ["Physically supported", "0"],
      ["Batteries reporting IDLING", "93 / 93"],
      ["Authority", "not controllable"],
    ],
  },
  {
    key: "statistical",
    label: "Statistical",
    value: "Available",
    tone: "available",
    body: "How far demand has historically moved from its own expected profile — estimated, calibrated on a conformity split, and evaluated once on held-out rows.",
    measures: [
      ["Coverage @ 90%", "0.914 / 0.911 / 0.893"],
      ["Mean width kW", "2.73 / 5.57 / 14.12"],
      ["Basis", "statistical proxy"],
      ["Dispatchable", "no"],
    ],
  },
  {
    key: "assumed",
    label: "Assumed",
    value: "Scenario only",
    tone: "scenario",
    body: "A stated assumption an operator chooses to plan against. Recorded as an assumption, never merged into a measurement and never presented as capacity.",
    measures: [
      ["Basis", "operator assumption"],
      ["Merged with measurement", "never"],
      ["Presented as capacity", "no"],
      ["Aggregation creates capability", "no"],
    ],
  },
];

export const FLEX_HORIZONS = ["h=1 · 15 min", "h=4 · 1 h", "h=96 · 24 h"];

/** Section — the research ledger. `energy-intelligence console status`. */
export interface LedgerPhase {
  n: number;
  name: string;
  verdict: "COMPLETE" | "NEGATIVE" | "YES" | "NO" | "SEE REPORT";
  headline: string;
  note: string;
}

export const PHASE_LEDGER: LedgerPhase[] = [
  {
    n: 6,
    name: "Qwen specialization",
    verdict: "NEGATIVE",
    headline: "underperformed; recorded not engineered around",
    note: "A frozen Qwen3-1.7B trunk with a task head lost to the classical GBM at every horizon: 2.74 vs 1.56 kW MAE at 24 h on identical rows. The miss was pre-registered before the study ran.",
  },
  {
    n: 7,
    name: "Heterogeneous expert router",
    verdict: "NEGATIVE",
    headline: "router did not beat the fixed ensemble",
    note: "The diversity premise held — experts disagree on 59–75% of rows — but the learned gate lost at 3 of 3 horizons. The fixed weighted ensemble ships instead.",
  },
  {
    n: 8,
    name: "Calibrated uncertainty",
    verdict: "YES",
    headline: "coverage within 0.7pp at 90%, all horizons",
    note: "Six interval methods fitted, calibrated and scored once on the sealed test split. The default ±1σ interval failed 11 of 12 cells; that failure ships alongside the result.",
  },
  {
    n: 9,
    name: "Flexibility + capability audit",
    verdict: "NO",
    headline: "0 of 8 dimensions physically supported",
    note: "A behavioural envelope was estimated and published as statistical proxy / not controllable. No physical flexibility exists in this dataset, so no dispatchable capability is claimed.",
  },
  {
    n: 10,
    name: "Controlled feature ablation",
    verdict: "SEE REPORT",
    headline: "feature set is the only variable",
    note: "Model, folds, seed and horizon held fixed while feature families changed. Load's selected set did not survive confirmation, and that is reported rather than hidden.",
  },
];

/** Section — provenance chain. Each `evidence` value is a real artifact. */
export interface ProvenanceStep {
  key: string;
  label: string;
  detail: string;
  evidence: string;
  evidenceTitle: string;
}

export const PROVENANCE: ProvenanceStep[] = [
  {
    key: "data",
    label: "Data",
    detail:
      "SMART-DS v1.0 ingested file by file, normalized to kW, validated against the dataset's own published figures before any model was fit.",
    evidence: "ds-16ff1dabe80b",
    evidenceTitle: "dataset version · 359 files SHA-256'd",
  },
  {
    key: "model",
    label: "Model",
    detail:
      "Configuration frozen before fit: feature list, hyperparameters, seed and horizon are recorded and checksummed, not chosen after the fact.",
    evidence: "c8fe83d1b9e9b561",
    evidenceTitle: "model config checksum · frozen before fit",
  },
  {
    key: "experiment",
    label: "Experiment",
    detail:
      "The protocol is hashed and committed before any result is read, so the rules a run played by cannot be edited to fit its outcome.",
    evidence: "472ff9c1b723787f",
    evidenceTitle: "protocol freeze hash · 30 runs registered",
  },
  {
    key: "evidence",
    label: "Evidence",
    detail:
      "Artifacts are retained per cell — timestamps, predictions, errors — so every published figure can be recomputed from its own source rows.",
    evidence: "960fcc75cd819c1d",
    evidenceTitle: "selection freeze hash · evidence registry",
  },
  {
    key: "decision",
    label: "Decision",
    detail:
      "A claim moves only when its evidence does. Negative results are recorded at the same rank as positive ones.",
    evidence: "D-001 → D-130",
    evidenceTitle: "decision records, alternatives and consequences",
  },
];

/** Truncated display form of the hashes above, for the small print. */
export const PROVENANCE_FOOTNOTE =
  "dataset 7e860b26ffbf… · protocol 472ff9c1b723787f… · freeze 960fcc75cd819c1d…";

/** Hero metadata rail. */
export const HERO_META = [
  "SMART-DS",
  "15-min native grid",
  "H24 forecasting",
  "Evidence-driven",
];

/** Section 05 — the escalation from uncertainty to planning posture. */
export const ESCALATION = [
  { from: "More uncertainty", to: "Less confidence" },
  { from: "Less confidence", to: "More conservative planning" },
];
