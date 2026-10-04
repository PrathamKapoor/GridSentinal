import { Link } from "react-router-dom";
import Grainient, { INDUSTRIAL_PARAMS } from "../../components/Grainient";
import { LOOP } from "../../data/system";
import "./Hero.css";

const STRIP_FACTS = [
  ["Phases complete", "1–8 of 20"],
  ["Test suite", "1,235 passing"],
  ["Telemetry", "35,040 × 15 min"],
  ["Experiments recorded", "29"],
  ["Decision records", "D-001…D-109"],
];

const LOOP_PREVIEW = LOOP.slice(0, 8);

export default function Hero() {
  return (
    <section className="hero" aria-label="GridSentinal introduction">
      <div className="hero-atmosphere" aria-hidden="true">
        <Grainient params={INDUSTRIAL_PARAMS} className="hero-grainient" />
        <div className="hero-atmosphere-fade" />
      </div>

      <div className="container hero-inner">
        <p className="hero-eyebrow mono">
          Adaptive · self-verifying energy management
        </p>

        <h1 className="hero-title">
          Energy intelligence that knows{" "}
          <span className="hero-title-accent">when to act.</span>
        </h1>

        <p className="hero-lede">
          GridSentinal observes an energy system, forecasts what happens next,
          and states how confident it is. Proposed actions are attacked,
          simulated and verified before they execute — so that decisions rest on
          evidence, not on a confident-looking prediction.
        </p>

        <div className="hero-ctas">
          <Link to="/console" className="btn btn-primary">
            Enter GridSentinal
            <svg width="13" height="12" viewBox="0 0 13 12" fill="none" aria-hidden="true">
              <path
                d="M1 6h10M7.4 2.2 11.2 6 7.4 9.8"
                stroke="currentColor"
                strokeWidth="1.5"
              />
            </svg>
          </Link>
          <a href="#system" className="btn btn-secondary">
            Explore the system
          </a>
        </div>

        <dl className="hero-strip" aria-label="Recorded project facts">
          {STRIP_FACTS.map(([label, value]) => (
            <div className="hero-strip-cell" key={label}>
              <dt>{label}</dt>
              <dd className="mono">{value}</dd>
            </div>
          ))}
        </dl>

        <div className="hero-loop" aria-hidden="true">
          {LOOP_PREVIEW.map((s, i) => (
            <span className="hero-loop-item" key={s.key}>
              <span
                className={`hero-loop-dot ${
                  s.status === "operational" ? "is-live" : ""
                }`}
              />
              <span
                className={`hero-loop-label mono ${
                  s.status === "operational" ? "is-live" : ""
                }`}
              >
                {s.label}
              </span>
              {i < LOOP_PREVIEW.length - 1 && (
                <span className="hero-loop-line" />
              )}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
