import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import ForecastChart from "../../components/ForecastChart";
import evidence from "../../data/evidence.json";
import { DATASET_FACTS } from "../../data/system";
import "./ProductPreview.css";

interface ForecastSlice {
  asset_id: string;
  horizon_label: string;
  method: string;
  timestamps: string[];
  point_kw: number[];
  lower_kw: number[];
  upper_kw: number[];
}

interface Procedure {
  horizon: number;
  method: string;
  coverage_90: number;
  coverage_error: number;
  spearman_width_error: number;
  top_decile_multiple: number;
  mean_width_kw: number;
}

const slice = evidence.forecast_slice as ForecastSlice;
// the slice is the 1-hour-ahead horizon (h = 4 steps of 15 minutes)
const procedure = (
  evidence.published_procedure as Procedure[]
).find((p) => p.horizon === 4)!;

const PIPELINE_STEPS = [
  ["Propose", "optimizer · in design", "locked"],
  ["Simulate", "digital twin · in design", "locked"],
  ["Challenge", "red team · in design", "locked"],
  ["Verify", "assurance gate · in design", "locked"],
] as const;

export default function ProductPreview() {
  const first = slice.timestamps[0]!.slice(0, 10);
  const last = slice.timestamps[slice.timestamps.length - 1]!.slice(0, 10);

  const systemRows: [string, string][] = [
    ["Dataset", DATASET_FACTS.dataset],
    ["Feeder", DATASET_FACTS.feeder],
    ["Nodes", DATASET_FACTS.nodes],
    ["Customer loads", DATASET_FACTS.customers],
    ["PV systems", DATASET_FACTS.pvSystems],
    ["Batteries", DATASET_FACTS.batteries],
    ["Interval", DATASET_FACTS.interval],
  ];

  const qualityRows: [string, string][] = [
    ["Coverage @ 90%", `${(procedure.coverage_90 * 100).toFixed(1)}%`],
    ["Coverage error", `${procedure.coverage_error > 0 ? "+" : ""}${(procedure.coverage_error * 100).toFixed(2)} pp`],
    ["Mean width", `${procedure.mean_width_kw.toFixed(2)} kW`],
    ["ρ width ↔ error", `+${procedure.spearman_width_error.toFixed(3)}`],
    ["Widest decile", `${procedure.top_decile_multiple.toFixed(2)}× mean error`],
  ];

  return (
    <section className="section" id="product">
      <div className="container">
        <SectionHead
          index="03"
          kicker="The product"
          title="What the system sees."
          lede="A command center for one feeder: live demand and generation, a forecast that carries its own calibrated interval, and a decision pipeline that shows why an action is allowed to move. Everything below is rendered from the repository's recorded experiment output."
        />

        <Reveal rise={40}>
          <div className="console">
            <div className="console-chrome">
            <span className="console-crumb mono">
              GRIDSENTINAL <span className="console-crumb-sep">/</span>{" "}
              FEEDER P1UHS0_1247
            </span>
            <div className="console-chips">
              <span className="chip chip--warn">
                <span className="chip-dot" />
                demonstration state
              </span>
              <span className="chip chip--accent">
                <span className="chip-dot" />
                intervals · {slice.method} @ 90%
              </span>
              <span className="chip">telemetry · 15 min</span>
            </div>
          </div>

          <div className="console-body">
            <aside className="console-col">
              <div className="panel">
                <div className="panel-head">
                  <span className="panel-title">System</span>
                </div>
                <dl className="kv">
                  {systemRows.map(([k, v]) => (
                    <div className="kv-row" key={k}>
                      <dt>{k}</dt>
                      <dd className="mono">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <span className="panel-title">Forecast</span>
                </div>
                <dl className="kv">
                  <div className="kv-row">
                    <dt>Point model</dt>
                    <dd className="mono">fixed ensemble</dd>
                  </div>
                  <div className="kv-row">
                    <dt>Horizon</dt>
                    <dd className="mono">{slice.horizon_label}</dd>
                  </div>
                  <div className="kv-row">
                    <dt>Asset</dt>
                    <dd className="mono">{slice.asset_id.replace("asset-", "")}</dd>
                  </div>
                  <div className="kv-row">
                    <dt>Window</dt>
                    <dd className="mono">
                      {first} → {last}
                    </dd>
                  </div>
                </dl>
              </div>
            </aside>

            <div className="panel console-chart">
              <div className="panel-head">
                <span className="panel-title">Demand forecast · kW</span>
                <span className="console-chart-note mono">
                  point + calibrated 90% interval
                </span>
              </div>
              <div className="console-chart-body">
                <ForecastChart
                  series={{
                    timestamps: slice.timestamps,
                    point: slice.point_kw,
                    lower: slice.lower_kw,
                    upper: slice.upper_kw,
                  }}
                  caption={`One-hour-ahead demand forecast for asset ${slice.asset_id}, recorded model output`}
                />
              </div>
            </div>

            <aside className="console-col">
              <div className="panel">
                <div className="panel-head">
                  <span className="panel-title">Interval quality</span>
                  <span className="chip chip--ok chip-sm">
                    <span className="chip-dot" />
                    calibrated
                  </span>
                </div>
                <dl className="kv">
                  {qualityRows.map(([k, v]) => (
                    <div className="kv-row" key={k}>
                      <dt>{k}</dt>
                      <dd className="mono">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <span className="panel-title">Decision pipeline</span>
                </div>
                <div className="pipeline">
                  {PIPELINE_STEPS.map(([label, phase]) => (
                    <div className="pipeline-step" key={label}>
                      <span className="pipeline-lock" aria-hidden="true">
                        <svg width="9" height="11" viewBox="0 0 9 11" fill="none">
                          <rect x="1" y="4.6" width="7" height="5.4" rx="1" stroke="currentColor" strokeWidth="1.1" />
                          <path d="M2.6 4.6V3a1.9 1.9 0 0 1 3.8 0v1.6" stroke="currentColor" strokeWidth="1.1" />
                        </svg>
                      </span>
                      <span className="pipeline-label">{label}</span>
                      <span className="pipeline-phase mono">{phase}</span>
                    </div>
                  ))}
                  <p className="pipeline-empty">
                    No candidate actions yet; the optimizer is not implemented.
                    This is where a verified proposal will appear.
                  </p>
                </div>
              </div>
            </aside>
          </div>
          </div>
        </Reveal>

        <Reveal delay={160}>
          <p className="console-foot source-note">
            Demonstration console, not a live system. The chart is recorded
            model output (conformal_state, sealed test split) on real SMART-DS
            telemetry; system figures come from the verified ingestion report.
            The full command center is still under construction.
          </p>
        </Reveal>
      </div>
    </section>
  );
}
