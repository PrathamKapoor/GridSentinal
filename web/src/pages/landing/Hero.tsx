import { Link } from "react-router-dom";
import Grainient, { INDUSTRIAL_PARAMS } from "../../components/Grainient";
import Marquee from "../../components/anim/Marquee";
import ForecastChart from "../../components/ForecastChart";
import evidence from "../../data/evidence.json";
import { LOOP } from "../../data/system";
import "./Hero.css";

interface ForecastSlice {
  timestamps: string[];
  point_kw: number[];
  lower_kw: number[];
  upper_kw: number[];
}

const slice = evidence.forecast_slice as ForecastSlice;

const TICKER = [
  "35,040 telemetry points",
  "1,235 tests passing",
  "Phases 1–8 of 20 complete",
  "Sealed test split · read once",
  "29 experiments registered",
  "SMART-DS feeder · 2018",
  "90% calibrated intervals",
  "No claim without evidence",
];

const LOOP_PREVIEW = LOOP.slice(0, 6);

const GREEN_PARAMS = {
  ...INDUSTRIAL_PARAMS,
  colorA: [0.008, 0.022, 0.013] as [number, number, number],
  colorB: [0.03, 0.09, 0.05] as [number, number, number],
  colorC: [0.16, 0.55, 0.3] as [number, number, number],
  colorD: [0.1, 0.45, 0.42] as [number, number, number],
  energy: 0.6,
  vignette: 0.8,
};

export default function Hero() {
  return (
    <section className="hero" aria-label="GridSentinal introduction">
      <div className="hero-atmosphere" aria-hidden="true">
        <Grainient params={GREEN_PARAMS} className="hero-grainient" />
        <div className="hero-atmosphere-fade" />
        <div className="hero-glow" />
      </div>

      <div className="container hero-inner">
        <p className="hero-badge" data-hero-item style={{ "--i": 0 } as React.CSSProperties}>
          <span className="hero-badge-dot" aria-hidden="true" />
          Adaptive · self-verifying energy management
        </p>

        <h1 className="hero-title">
          <span className="hero-line" data-hero-item style={{ "--i": 1 } as React.CSSProperties}>
            Energy intelligence
          </span>
          <span className="hero-line" data-hero-item style={{ "--i": 2 } as React.CSSProperties}>
            that knows <em>when to act.</em>
          </span>
        </h1>

        <p className="hero-lede" data-hero-item style={{ "--i": 3 } as React.CSSProperties}>
          GridSentinal observes an energy system, forecasts what happens next,
          and states how confident it is. Proposed actions are attacked,
          simulated and verified before they execute — evidence, not optimism.
        </p>

        <div className="hero-ctas" data-hero-item style={{ "--i": 4 } as React.CSSProperties}>
          <Link to="/console" className="btn btn-primary">
            Enter GridSentinal
            <svg className="btn-arrow" width="14" height="12" viewBox="0 0 14 12" fill="none" aria-hidden="true">
              <path d="M1 6h11M7.8 1.8 12 6l-4.2 4.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </Link>
          <a href="#system" className="btn btn-secondary">
            Explore the system
          </a>
        </div>
      </div>

      {/* live signal card - recorded Phase 8 output, rises into view */}
      <div className="container hero-card-wrap" data-hero-item style={{ "--i": 5 } as React.CSSProperties}>
        <div className="hero-card" aria-label="Recorded forecast preview">
          <div className="hero-card-head">
            <span className="hero-card-title mono">DEMAND · 1 H AHEAD · 90% INTERVAL</span>
            <span className="chip chip--ok chip-sm">
              <span className="chip-dot" />
              calibrated
            </span>
          </div>
          <div className="hero-card-chart chart-appear">
            <ForecastChart
              series={{
                timestamps: slice.timestamps,
                point: slice.point_kw,
                lower: slice.lower_kw,
                upper: slice.upper_kw,
              }}
              caption="One-hour-ahead demand forecast with calibrated 90 percent interval, recorded Phase 8 output"
              height={230}
            />
          </div>
          <p className="hero-card-foot mono">
            recorded Phase 8 output · conformal_state · SMART-DS 2018 · not a live system
          </p>
        </div>
      </div>

      {/* full-bleed fact ticker */}
      <div className="hero-ticker" data-hero-item style={{ "--i": 6 } as React.CSSProperties}>
        <Marquee
          items={TICKER}
          speed={42}
          ariaLabel={TICKER.join(", ")}
          className="hero-ticker-marquee"
        />
      </div>

      {/* loop preview rail */}
      <div className="container">
        <div className="hero-loop" aria-hidden="true">
          {LOOP_PREVIEW.map((s, i) => (
            <span className="hero-loop-item" key={s.key}>
              <span
                className={`hero-loop-dot ${s.status === "operational" ? "is-live" : ""}`}
                style={{ animationDelay: `${i * 0.5}s` }}
              />
              <span className={`hero-loop-label mono ${s.status === "operational" ? "is-live" : ""}`}>
                {s.label}
              </span>
              {i < LOOP_PREVIEW.length - 1 && <span className="hero-loop-line" />}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
