import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/anim/Reveal";
import Logo from "../components/Logo";
import { DATASET_FACTS } from "../data/system";
import "./Console.css";

type SubsystemStatus = "operational" | "partial" | "not implemented";

interface Subsystem {
  name: string;
  status: SubsystemStatus;
  layer: string;
  detail: string;
  facts?: [string, string][];
}

const SUBSYSTEMS: Subsystem[] = [
  {
    name: "Data & telemetry",
    status: "operational",
    layer: "ingestion",
    detail:
      "SMART-DS v1.0 ingested, normalized and reality-validated. Every file SHA-256 checksummed; balance identities that fail are reported, not adjusted.",
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
    layer: "modelling",
    detail:
      "Fixed weighted ensemble over three heterogeneous experts, chosen because it beat the learned router at all horizons. The point forecast is pinned to the recorded ensemble artifact and may not be refitted.",
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
    layer: "calibration",
    detail:
      "Six interval methods fitted and scored once on the sealed split. Coverage within ±0.01 of nominal at every published horizon. No coverage guarantee is claimed.",
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
    layer: "optimization",
    detail:
      "The optimizer that converts forecasts, uncertainty, flexibility and constraints into candidate actions. Nothing here yet; no mock actions are shown.",
  },
  {
    name: "Digital twin",
    status: "not implemented",
    layer: "simulation",
    detail:
      "Independent simulation of proposed actions. Blocked on a loss model (G-06); per-node power balance must be produced, not read.",
  },
  {
    name: "Red team & assurance",
    status: "not implemented",
    layer: "verification",
    detail:
      "Adversarial challenges and the decision-assurance gate. The scoring function is deliberately undefined until the layer that owns it.",
  },
  {
    name: "Flexibility",
    status: "partial",
    layer: "modelling",
    detail:
      "Representation exists in the domain (FlexibilityEstimate). Battery dispatch is unavailable in the dataset (G-01), a data blocker that is recorded, not simulated.",
  },
  {
    name: "Energy-aware MLOps",
    status: "not implemented",
    layer: "lifecycle",
    detail:
      "Drift detection, retraining and deployment gates. The discipline they will automate already runs by hand.",
  },
];

const STATS: [string, ReactNode][] = [
  ["Loop stages live", "4 of 11"],
  ["Tests passing", "1,235"],
  ["Experiments registered", "29"],
  ["Telemetry points", "35,040"],
];

function StatusChip({ status }: { status: SubsystemStatus }) {
  return (
    <span className={`cstatus cstatus--${status.replace(" ", "-")}`}>
      <span className="cstatus-dot" aria-hidden="true" />
      {status}
    </span>
  );
}

export default function Console() {
  return (
    <div className="cpage">
      <a href="#cmain" className="cskip">
        Skip to content
      </a>

      <header className="ctop">
        <div className="ctop-inner">
          <Link to="/" className="cbrand">
            <Logo size={20} />
            <span>GridSentinal</span>
            <span className="cbrand-loc">console</span>
          </Link>
          <nav className="ctop-nav" aria-label="Console">
            <Link to="/" className="ctop-link">
              ← Back to the system
            </Link>
          </nav>
        </div>
      </header>

      <main id="cmain" className="cmain">
        <div className="ccontainer">
          <Reveal>
            <header className="cheader">
              <p className="clabel">GridSentinal · Entry point</p>
              <h1 className="ctitle">System console</h1>
              <p className="cdesc">
                This console reports the system's recorded state and nothing
                else. Subsystems marked <strong>not implemented</strong> have
                no mock data, no simulated dashboards and no fake numbers;
                they are listed because the architecture names them, and the
                console does not pretend otherwise.
              </p>
            </header>
          </Reveal>

          <Reveal delay={80}>
            <dl className="cstats">
              {STATS.map(([label, value]) => (
                <div className="cstat" key={label}>
                  <dt className="clabel">{label}</dt>
                  <dd className="cstat-value num">{value}</dd>
                </div>
              ))}
            </dl>
          </Reveal>

          <Reveal delay={120}>
            <section aria-labelledby="csubs">
              <h2 id="csubs" className="clabel csection-label">
                Subsystems
              </h2>
              <div className="ctable" role="table" aria-label="Subsystem status">
                {SUBSYSTEMS.map((s, i) => (
                  <Reveal
                    key={s.name}
                    delay={Math.min(i * 55, 440)}
                    className="crow"
                    role="row"
                  >
                    <div className="crow-name" role="cell">
                      <span className="crow-title">{s.name}</span>
                      <span className="crow-phase mono">{s.layer}</span>
                    </div>
                    <div className="crow-status" role="cell">
                      <StatusChip status={s.status} />
                    </div>
                    <div className="crow-detail" role="cell">
                      {s.detail}
                    </div>
                  </Reveal>
                ))}
              </div>
            </section>
          </Reveal>

          <Reveal delay={80}>
            <section aria-labelledby="crec">
              <h2 id="crec" className="clabel csection-label">
                Recorded state · operational layers
              </h2>
              <div className="ccards">
                {SUBSYSTEMS.filter((s) => s.facts).map((s, i) => (
                  <Reveal
                    key={s.name}
                    delay={i * 110}
                    className="ccard"
                    as="article"
                  >
                    <div className="ccard-head">
                      <h3 className="ccard-title">{s.name}</h3>
                      <StatusChip status={s.status} />
                    </div>
                    <p className="ccard-detail">{s.detail}</p>
                    <dl className="cfacts">
                      {s.facts!.map(([k, v]) => (
                        <div className="cfact" key={k}>
                          <dt>{k}</dt>
                          <dd className="mono num">{v}</dd>
                        </div>
                      ))}
                    </dl>
                  </Reveal>
                ))}
              </div>
            </section>
          </Reveal>

          <Reveal delay={120}>
            <p className="cnote">
              The command center, with live forecasts, decision pipelines and
              simulation runs, is still under construction. Reproduce
              everything shown here from the repository:{" "}
              <code className="mono">
                energy-intel uncertainty experiments
              </code>{" "}
              prints the recorded calibration result.
            </p>
          </Reveal>
        </div>
      </main>

      <footer className="cfoot">
        <div className="ccontainer cfoot-inner">
          <span>
            Research system. No production deployment, no customers, no uptime
            claims; the evidence is the repository.
          </span>
          <span className="cfoot-license mono">
            Proprietary. All rights reserved.
          </span>
        </div>
      </footer>
    </div>
  );
}
