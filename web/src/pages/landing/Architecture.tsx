import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import "./Architecture.css";

interface ComponentRow {
  name: string;
  detail: string;
  status: "operational" | "partial" | "representation" | "specified";
}

const STACK: ComponentRow[] = [
  {
    name: "Physical system",
    detail: "SMART-DS v1.0 feeder, AUS/P1U 2018: the modelled reality.",
    status: "operational",
  },
  {
    name: "Telemetry & data quality",
    detail: "359 files SHA-256'd, normalized to kW, validated against the dataset's own published figures.",
    status: "operational",
  },
  {
    name: "Energy system model",
    detail: "Formal contract: state, assets, topology, constraints, provenance, quality.",
    status: "operational",
  },
  {
    name: "Forecasting & regime",
    detail: "Ensemble of persistence, GBM, temporal TCN, routed by measured regime structure.",
    status: "operational",
  },
  {
    name: "Uncertainty model",
    detail: "Conformal calibration around the fixed ensemble; the published procedure.",
    status: "operational",
  },
  {
    name: "Flexibility engine",
    detail: "Battery dispatch is not derivable from the dataset; representation only, blocker recorded.",
    status: "representation",
  },
  {
    name: "Optimization",
    detail: "Forecasts + uncertainty + flexibility + constraints → candidate actions.",
    status: "specified",
  },
  {
    name: "Digital twin",
    detail: "Independent simulation of a proposed action, producing a comparable outcome.",
    status: "specified",
  },
  {
    name: "Red team",
    detail: "Adversarial scenarios probing when a decision becomes unsafe or suboptimal.",
    status: "specified",
  },
  {
    name: "Decision assurance",
    detail: "The gate between proposal and execution. Approve, or reject and re-optimize.",
    status: "specified",
  },
  {
    name: "Energy-aware MLOps",
    detail: "Drift, retraining, evaluation and deployment gates, supervised by agents.",
    status: "specified",
  },
];

const STATUS_CHIP: Record<ComponentRow["status"], string> = {
  operational: "chip--ok",
  partial: "chip--warn",
  representation: "chip--warn",
  specified: "chip--accent",
};

const STATUS_LABEL: Record<ComponentRow["status"], string> = {
  operational: "operational",
  partial: "partial",
  representation: "representation only",
  specified: "in design",
};

export default function Architecture() {
  return (
    <section className="section" id="architecture">
      <div className="container">
        <SectionHead
          index="11"
          kicker="Architecture"
          title="How the components relate."
          lede="The target architecture, with each boundary held explicit and each layer's real status. Nothing is marked operational that has not been run end to end against the sealed test split."
        />

        <div className="arch" role="list" aria-label="System architecture layers">
          {STACK.map((c, i) => (
            <Reveal key={c.name} delay={Math.min(i * 60, 480)} role="listitem" className="arch-row">
              <span className="arch-node" aria-hidden="true" />
              <div className="arch-name">
                <span className="arch-name-text mono">{c.name}</span>
              </div>
              <p className="arch-detail">{c.detail}</p>
              <div className="arch-status">
                <span className={`chip ${STATUS_CHIP[c.status]}`}>
                  <span className="chip-dot" />
                  {STATUS_LABEL[c.status]}
                </span>
              </div>
            </Reveal>
          ))}
          <Reveal delay={200}>
            <div className="arch-loopback mono" aria-hidden="true">
              ↺ outcomes return to telemetry; the loop closes
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
