import { Link } from "react-router-dom";
import Grainient, { INDUSTRIAL_PARAMS } from "../../components/Grainient";
import "./FinalCta.css";

const CTA_PARAMS = {
  ...INDUSTRIAL_PARAMS,
  energy: 0.62,
  vignette: 0.85,
};

export default function FinalCta() {
  return (
    <section className="final-cta" id="enter" aria-label="Enter GridSentinal">
      <div className="final-cta-atmosphere" aria-hidden="true">
        <Grainient params={CTA_PARAMS} />
        <div className="final-cta-fade" />
      </div>
      <div className="container final-cta-inner">
        <p className="final-cta-kicker mono">08 · Enter</p>
        <h2 className="final-cta-title">
          See what the system sees.
        </h2>
        <p className="final-cta-lede">
          The console reports the system's real, recorded state — the ingested
          feeder, the fixed ensemble, the calibration results, the experiment
          registry, and everything still on the roadmap. No mock data presented
          as live.
        </p>
        <div className="final-cta-actions">
          <Link to="/console" className="btn btn-primary btn-lg">
            Enter GridSentinal
            <svg width="13" height="12" viewBox="0 0 13 12" fill="none" aria-hidden="true">
              <path
                d="M1 6h10M7.4 2.2 11.2 6 7.4 9.8"
                stroke="currentColor"
                strokeWidth="1.5"
              />
            </svg>
          </Link>
        </div>
      </div>
    </section>
  );
}
