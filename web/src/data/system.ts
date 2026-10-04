/**
 * Static, honest system facts.
 *
 * Every value here is a project fact recorded in the repository
 * (README.md, handoff.md, docs/, experiments/registry.jsonl). Nothing is
 * aspirational: unimplemented capabilities are listed as specified.
 */

export interface LoopStage {
  key: string;
  label: string;
  status: "operational" | "specified";
  owner: string;
  description: string;
}

export const LOOP: LoopStage[] = [
  {
    key: "observe",
    label: "Observe",
    status: "operational",
    owner: "telemetry pipeline",
    description:
      "SMART-DS telemetry ingested, normalized and validated against the dataset's own published figures: 5,196 nodes, 1,871 customer loads, 35,040 fifteen-minute points, every file SHA-256 checksummed.",
  },
  {
    key: "understand",
    label: "Understand",
    status: "operational",
    owner: "energy system model",
    description:
      "A formal energy-system contract (state, assets, topology, provenance, quality) plus measured regime structure: error concentrates in high-demand and high-ramp periods, quantified before any model was fit.",
  },
  {
    key: "predict",
    label: "Predict",
    status: "operational",
    owner: "forecast ensemble",
    description:
      "Ensemble forecasting at 15 min / 1 h / 24 h horizons. Persistence, gradient boosting, a temporal TCN and a learned router were all evaluated on identical sealed rows; the fixed weighted ensemble ships.",
  },
  {
    key: "quantify",
    label: "Quantify uncertainty",
    status: "operational",
    owner: "calibration study",
    description:
      "Six interval methods fitted, calibrated and scored once on a sealed test split. The published procedure knows how confident it is, and says so honestly.",
  },
  {
    key: "options",
    label: "Generate options",
    status: "specified",
    owner: "decision engine",
    description:
      "The optimizer converts forecast, uncertainty, flexibility and constraints into candidate actions. Not implemented: awaits the decision engine.",
  },
  {
    key: "attack",
    label: "Attack",
    status: "specified",
    owner: "red team",
    description:
      "Red-team scenarios deliberately try to break each candidate: when does it become unsafe, ineffective or suboptimal?",
  },
  {
    key: "simulate",
    label: "Simulate",
    status: "specified",
    owner: "digital twin",
    description:
      "A digital twin simulates each proposed action independently of the learned model, producing an outcome to compare against the prediction.",
  },
  {
    key: "verify",
    label: "Verify",
    status: "specified",
    owner: "assurance gate",
    description:
      "A decision-assurance gate between proposal and execution: approve or reject, with adaptive autonomy. The scoring function is deliberately undefined until the layer that owns it.",
  },
  {
    key: "decide",
    label: "Decide",
    status: "specified",
    owner: "assurance gate",
    description:
      "Execute only when evidence supports the action; otherwise reject and re-optimize. Every consequential action is confirmed.",
  },
  {
    key: "act",
    label: "Act & observe result",
    status: "specified",
    owner: "execution layer",
    description:
      "Executed actions return outcomes to the system as performance data, closing the loop.",
  },
  {
    key: "learn",
    label: "Learn",
    status: "specified",
    owner: "mlops",
    description:
      "Energy-aware MLOps: drift detection, retraining and deployment gates keep the learned model honest over time.",
  },
];

/** Dataset facts, from the verified ingestion report. */
export const DATASET_FACTS = {
  dataset: "SMART-DS v1.0",
  region: "AUS / P1U, 2018",
  feeder: "p1uhs0_1247–p1udt12703",
  nodes: "5,196",
  customers: "1,871",
  edges: "4,555",
  pvSystems: "1,216",
  batteries: "93",
  interval: "15 min",
  points: "35,040",
  files: "359",
  size: "228.5 MiB",
  checksums: "SHA-256 per file",
};

/** Process facts that make the work checkable. */
export const PROCESS_FACTS = {
  tests: "1,356",
  experiments: 30,
  decisionRecords: "D-130",
  loopStagesLive: 4,
  loopStagesTotal: LOOP.length,
};
