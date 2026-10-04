import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import { FLEX_LAYERS, FLEX_HORIZONS } from "../../data/landing";
import "./Flexibility.css";

const COVERAGE = ["0.9140", "0.9110", "0.8925"];
const WIDTH = ["2.73", "5.57", "14.12"];

export default function Flexibility() {
  return (
    <section className="section flexibility" id="flexibility">
      <div className="container">
        <SectionHead
          index="05"
          kicker="Honest flexibility"
          title={
            <>
              Flexibility
              <br />
              without <em>fiction.</em>
            </>
          }
          lede="Observed demand behaviour is not automatically controllable capacity. GridSentinal keeps three different claims in three different boxes, because collapsing them into one number is how a forecast quietly becomes an instruction."
        />

        <div className="flex-layers">
          {FLEX_LAYERS.map((l, i) => (
            <Reveal
              className="flex-layer"
              key={l.key}
              delay={i * 120}
              as="article"
            >
              <div className={`flex-card flex-card--${l.tone}`}>
                <div className="flex-card-head">
                  <span className="flex-label mono">{l.label}</span>
                  <span
                    className={`chip chip--${
                      l.tone === "available" ? "ok" : l.tone === "scenario" ? "warn" : "fail"
                    }`}
                  >
                    <span className="chip-dot" />
                    {l.value}
                  </span>
                </div>

                <p className="flex-body">{l.body}</p>

                <dl className="flex-measures">
                  {l.measures.map(([k, v]) => (
                    <div className="flex-measure" key={k}>
                      <dt>{k}</dt>
                      <dd className="mono">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </Reveal>
          ))}
        </div>

        {/* the statistical layer, measured */}
        <Reveal delay={80} className="flex-measured">
          <div className="panel flex-panel">
            <div className="panel-head">
              <span className="panel-title">
                Behavioural envelope · sealed test split · 179,520 rows
              </span>
              <span className="chip chip--warn">
                <span className="chip-dot" />
                statistical proxy · not controllable
              </span>
            </div>

            <div className="flex-strip" role="list" aria-label="Envelope coverage by horizon">
              {FLEX_HORIZONS.map((h, i) => (
                <div className="flex-strip-item" role="listitem" key={h}>
                  <span className="flex-strip-h mono">{h}</span>
                  <span className="flex-strip-value mono">{COVERAGE[i]}</span>
                  <span className="flex-strip-label">coverage @ 90%</span>
                  <span className="flex-strip-sub mono">
                    mean width {WIDTH[i]} kW
                  </span>
                </div>
              ))}
            </div>

            <p className="table-note">
              Coverage sits within 0.0100 of nominal at every horizon, fitted on a
              conformity split and applied unchanged to held-out rows. Every figure
              carries{" "}
              <span className="mono">basis=STATISTICAL_PROXY</span> and{" "}
              <span className="mono">authority=NOT_CONTROLLABLE</span>. It describes
              how far demand has historically moved — not what an operator can
              command.
            </p>
          </div>
        </Reveal>

        <Reveal delay={140}>
          <div className="flex-caveat">
            <div className="flex-caveat-mark mono" aria-hidden="true">
              !
            </div>
            <p>
              <strong>Not claimed:</strong> dispatchable capacity, guaranteed demand
              response, or that aggregation creates authority. The eight audited
              dimensions are reported as <em>unknown</em> rather than as zero — the
              dataset holds no battery state-of-charge series, no enrolled response
              programme, and no control interface for any asset. That is a property
              of the data, and saying so is the point.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
