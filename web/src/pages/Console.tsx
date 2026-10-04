import { Link } from "react-router-dom";
import Nav from "../components/Nav";
import Footer from "../components/Footer";
import { DATASET_FACTS, PROCESS_FACTS } from "../data/system";
import "./Console.css";

type SubsystemStatus = "operational" | "partial" | "not implemented";

interface Subsystem {
  name: string;
  status: SubsystemStatus;
  phase: string;
  detail: string;
  facts?: [string, string][];
}

const SUBSYSTEMS: Subsystem[] = [
  {
    name: "Data & telemetry",
    status: "operational",
    phase: "Phase 3",
    detail: "SMART-DS v1.0 ingested, normalized and reality-validated. Every file SHA-256 checksummed; balance identities that fail are reported, not adjusted.",
    facts: [
      ["Dataset", DATASET_FACTS.dataset],
      ["Feeder", DATASET_FACTS.feeder],
      ["Nodes", DATASET_FACTS.nodes],
      ["Customer loads", DATASET_FACTS.customers],
      ["Telemetry", `${DATASET_FACTS.points} × 15 min`],
      ["Files", `${DATASET_FACTS.files} · ${DATASET_FACTS.size}`],
    ],
  },
  {
    name: "Point forecast",
    status: "operational",
    phase: "Phases 4–7",
    detail: "Fixed weighted ensemble over three heterogeneous experts, chosen because it beat the learned router at all horizons. The point forecast is pinned to Phase 7's artifact and may not be refitted.",
    facts: [
      ["Ensemble", "persistence + GBM + TCN"],
      ["Test MAE", "0.40 / 0.79 / 1.54 kW"],
      ["Horizons", "15 min · 1 h · 24 h"],
      ["Eval rows", "179,520"],
    ],
  },
  {
    name: "Calibrated uncertainty",
    status: "operational",
    phase: "Phase 8",
    detail: "Six interval methods fitted and scored once on the sealed split. Coverage within ±0.01 of nominal at every published horizon. No coverage guarantee is claimed.",
    facts: [
      ["Published", "conformal_state · conformal_dispersion"],
      ["Coverage @90%", "90.2 / 89.7 / 89.3%"],
      ["ρ width↔error", "+0.709 / +0.673 / +0.658"],
      ["Constant-width", "fails 11 of 12 cells"],
    ],
  },
  {
    name: "Decision engine",
    status: "not implemented",
    phase: "Phase 10",
    detail: "The optimizer that converts forecasts, uncertainty, flexibility and constraints into candidate actions. Nothing here yet — no mock actions are shown.",
  },
  {
    name: "Digital twin",
    status: "not implemented",
    phase: "Phase 11",
    detail: "Independent simulation of proposed actions. Blocked on a loss model (G-06); per-node power balance must be produced, not read.",
  },
  {
    name: "Red team & assurance",
    status: "not implemented",
    phase: "Phases 12–13",
    detail: "Adversarial challenges and the decision-assurance gate. The scoring function is deliberately undefined until the phase that owns it.",
  },
  {
    name: "Flexibility",
    status: "partial",
    phase: "Phase 9",
    detail: "Representation exists in the domain (FlexibilityEstimate). Battery dispatch is unavailable in the dataset (G-01) — a data blocker, recorded, not simulated.",
  },
  {
    name: "Energy-aware MLOps",
    status: "not implemented",
    phase: "Phases 14–15",
    detail: "Drift detection, retraining and deployment gates. The discipline they will automate already runs by hand.",
  },
];

const STATUS_CHIP: Record<SubsystemStatus, string> = {
  operational: "chip--ok",
  partial: "chip--warn",
  "not implemented": "chip--fail",
};

export default function Console() {
  return (
    <>
      <Nav />
      <main id="main" className="console-page">
        <div className="container">
          <header className="console-header">
            <p className="kicker">GridSentinal · entry point</p>
            <h1 className="console-title">System console</h1>
            <p className="console-lede">
              This console reports the system's recorded state and nothing
              else. Subsystems marked{" "}
              <span className="chip chip--fail chip-inline">
                <span className="chip-dot" />
                not implemented
              </span>{" "}
              have no mock data, no simulated dashboards and no fake numbers —
              they are listed because the architecture names them, and the
              console does not pretend otherwise.
            </p>
            <div className="console-header-facts">
              <span className="chip chip--accent">
                <span className="chip-dot" />
                Phase {PROCESS_FACTS.phasesDone} / {PROCESS_FACTS.phasesTotal}
              </span>
              <span className="chip">
                <span className="chip-dot" />
                {PROCESS_FACTS.tests} tests passing
              </span>
              <span className="chip">
                <span className="chip-dot" />
                {PROCESS_FACTS.experiments} experiments registered
              </span>
            </div>
          </header>

          <div className="console-grid">
            {SUBSYSTEMS.map((s) => (
              <section className={`panel console-card`} key={s.name} aria-label={`${s.name} status`}>
                <div className="panel-head">
                  <span className="panel-title">{s.name}</span>
                  <span className={`chip ${STATUS_CHIP[s.status]}`}>
                    <span className="chip-dot" />
                    {s.status}
                  </span>
                </div>
                <div className="console-card-body">
                  <p className="console-card-phase mono">{s.phase}</p>
                  <p className="console-card-detail">{s.detail}</p>
                  {s.facts && (
                    <dl className="kv">
                      {s.facts.map(([k, v]) => (
                        <div className="kv-row" key={k}>
                          <dt>{k}</dt>
                          <dd className="mono">{v}</dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </div>
              </section>
            ))}
          </div>

          <p className="console-note source-note">
            The command center — live forecasts, decision pipelines, simulation
            runs — arrives with Phase 19 of the roadmap. Reproduce everything
            shown here from the repository:{" "}
            <code className="mono">energy-intel uncertainty experiments</code>{" "}
            prints the recorded Phase 8 result.
          </p>

          <div className="console-back">
            <Link to="/" className="btn btn-secondary btn-sm">
              ← Back to the system
            </Link>
          </div>
        </div>
      </main>
      <Footer />
    </>
  );
}
