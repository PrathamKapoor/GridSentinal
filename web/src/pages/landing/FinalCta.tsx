import { Link } from "react-router-dom";
import Grainient, { INDUSTRIAL_PARAMS } from "../../components/Grainient";
import Marquee from "../../components/anim/Marquee";
import Reveal from "../../components/anim/Reveal";
import "./FinalCta.css";

const CTA_PARAMS = {
  ...INDUSTRIAL_PARAMS,
  colorA: [0.008, 0.022, 0.013] as [number, number, number],
  colorB: [0.03, 0.09, 0.05] as [number, number, number],
  colorC: [0.16, 0.55, 0.3] as [number, number, number],
  colorD: [0.1, 0.45, 0.42] as [number, number, number],
  energy: 0.72,
  vignette: 0.85,
};

const TICKER = [
  "See what the system sees",
  "Enter GridSentinal",
  "Evidence, not optimism",
  "Verified before it acts",
];

export default function FinalCta() {
  return (
    <section className="final-cta" id="enter" aria-label="Enter GridSentinal">
      <div className="final-cta-atmosphere" aria-hidden="true">
        <Grainient params={CTA_PARAMS} className="final-cta-grainient" />
        <div className="final-cta-fade" />
      </div>

      <div className="final-cta-ticker" aria-hidden="true">
        <Marquee items={TICKER} speed={30} className="final-cta-marquee" />
      </div>

      <div className="container final-cta-inner">
        <Reveal delay={0}>
          <p className="final-cta-kicker mono">08 · Enter</p>
        </Reveal>
        <Reveal delay={110}>
          <h2 className="final-cta-title">
            See what the
            <br />
            system <em>sees.</em>
          </h2>
        </Reveal>
        <Reveal delay={230}>
          <p className="final-cta-lede">
            The console reports the system's real, recorded state — the
            ingested feeder, the fixed ensemble, the calibration results, the
            experiment registry, and everything still on the roadmap. No mock
            data presented as live.
          </p>
        </Reveal>
        <Reveal delay={360}>
          <div className="final-cta-actions">
            <Link to="/console" className="btn btn-primary btn-lg">
              Enter GridSentinal
              <svg className="btn-arrow" width="14" height="12" viewBox="0 0 14 12" fill="none" aria-hidden="true">
                <path d="M1 6h11M7.8 1.8 12 6l-4.2 4.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </Link>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
