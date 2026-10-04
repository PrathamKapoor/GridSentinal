/**
 * Static, honest system facts.
 *
 * Every value here is a project fact recorded in the repository
 * (README.md, handoff.md, docs/, experiments/registry.jsonl). Nothing is
 * aspirational: unimplemented capabilities are listed as specified, with
 * their target phase.
 */

export interface PhaseFact {
  id: number;
  name: string;
  status: "complete" | "partial" | "negative" | "mixed" | "representation" | "planned";
  summary: string;
}

export const PHASES: PhaseFact[] = [
  { id: 1, name: "Foundation", status: "complete", summary: "Project scaffolding, configuration, health checks." },
  { id: 2, name: "Energy system model", status: "complete", summary: "Formal energy contract: state, actions, constraints, topology, provenance." },
  { id: 3, name: "Dataset ingestion", status: "complete", summary: "SMART-DS ingestion, normalization, reality validation." },
  { id: 4, name: "Baselines", status: "complete", summary: "ML dataset + naive, classical and neural baselines, all measured." },
  { id: 5, name: "Temporal model", status: "mixed", summary: "Energy Demand Dynamics TCN: wins at 15 min and 1 h, loses at 24 h." },
  { id: 6, name: "Language-model specialization", status: "negative", summary: "Qwen3-1.7B-Base adaptation: worse than classical at every horizon. Recorded." },
  { id: 7, name: "Expert router", status: "negative", summary: "Learned MoE router: loses to a fixed ensemble at all 3 horizons. Recorded." },
  { id: 8, name: "Calibrated uncertainty", status: "complete", summary: "Six interval methods, conformal calibration, sealed evaluation. Published." },
  { id: 9, name: "Flexibility modelling", status: "representation", summary: "Battery dispatch unavailable in the dataset; representation only." },
  { id: 10, name: "Decision engine", status: "planned", summary: "Optimization: forecasts + uncertainty + constraints to candidate actions." },
  { id: 11, name: "Digital twin", status: "planned", summary: "Independent simulation of proposed actions." },
  { id: 12, name: "Red team", status: "planned", summary: "Adversarial scenarios attacking proposed decisions." },
  { id: 13, name: "Decision assurance", status: "planned", summary: "Verification gate between proposal and execution." },
  { id: 14, name: "Energy-aware MLOps", status: "planned", summary: "Drift, retraining, deployment gates." },
];

/** The intelligence loop stages. `phase` names what implements the stage. */
export interface LoopStage {
  key: string;
  label: string;
  status: "operational" | "specified";
  phase: string;
  description: string;
}

export const LOOP: LoopStage[] = [
  {
    key: "observe",
    label: "Observe",
    status: "operational",
    phase: "Phase 3",
    description:
      "SMART-DS telemetry ingested, normalized and validated against the dataset's own published figures: 5,196 nodes, 1,871 customer loads, 35,040 fifteen-minute points, every file SHA-256 checksummed.",
  },
  {
    key: "understand",
    label: "Understand",
    status: "operational",
    phase: "Phases 2 & 4",
    description:
      "A formal energy-system contract (state, assets, topology, provenance, quality) plus measured regime structure: error concentrates in high-demand and high-ramp periods, quantified before any model was fit.",
  },
  {
    key: "predict",
    label: "Predict",
    status: "operational",
    phase: "Phases 4-7",
    description:
      "Ensemble forecasting at 15 min / 1 h / 24 h horizons. Persistence, gradient boosting, a temporal TCN and a learned router were all evaluated on identical sealed rows; the fixed weighted ensemble ships.",
  },
  {
    key: "quantify",
    label: "Quantify uncertainty",
    status: "operational",
    phase: "Phase 8",
    description:
      "Six interval methods fitted, calibrated and scored once on a sealed test split. The published procedure knows how confident it is - and says so honestly.",
  },
  {
    key: "options",
    label: "Generate options",
    status: "specified",
    phase: "Phase 10",
    description:
      "The optimizer converts forecast, uncertainty, flexibility and constraints into candidate actions. Not implemented: awaits the decision engine.",
  },
  {
    key: "attack",
    label: "Attack",
    status: "specified",
    phase: "Phase 12",
    description:
      "Red-team scenarios deliberately try to break each candidate: when does it become unsafe, ineffective or suboptimal?",
  },
  {
    key: "simulate",
    label: "Simulate",
    status: "specified",
    phase: "Phase 11",
    description:
      "A digital twin simulates each proposed action independently of the learned model, producing an outcome to compare against the prediction.",
  },
  {
    key: "verify",
    label: "Verify",
    status: "specified",
    phase: "Phase 13",
    description:
      "A decision-assurance gate between proposal and execution: approve or reject, with adaptive autonomy. The scoring function is deliberately undefined until the phase that owns it.",
  },
  {
    key: "decide",
    label: "Decide",
    status: "specified",
    phase: "Phase 13",
    description:
      "Execute only when evidence supports the action; otherwise reject and re-optimize. Every consequential action is confirmed.",
  },
  {
    key: "act",
    label: "Act & observe result",
    status: "specified",
    phase: "Phase 16",
    description:
      "Executed actions return outcomes to the system as performance data, closing the loop.",
  },
  {
    key: "learn",
    label: "Learn",
    status: "specified",
    phase: "Phases 14-15",
    description:
      "Energy-aware MLOps: drift detection, retraining and deployment gates keep the learned model honest over time.",
  },
];

/** Dataset facts, from Phase 3's verified ingestion report. */
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
  tests: "1,235",
  experiments: 29,
  decisionRecords: "D-109",
  phasesDone: 8,
  phasesTotal: 20,
};
