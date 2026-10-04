import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import { EXPERTS, ENSEMBLE, EXPERT_FACTS } from "../../data/landing";
import "./Experts.css";

const HORIS = EXPERT_FACTS.horizons;

export default function Experts() {
  return (
    <section className="section experts" id="forecast">
      <div className="container">
        <SectionHead
          index="03"
          kicker="Heterogeneous forecasting"
          title={
            <>
              Three experts disagree.
              <br />
              The disagreement is the signal.
            </>
          }
          lede="GridSentinal does not assume one model is always right. Persistence, gradient boosting and a temporal TCN are evaluated on identical sealed rows, and the system reports where they part company instead of averaging the difference away."
        />

        {/* fork diagram: one question, three answers, one signal */}
        <Reveal className="experts-fork" rise={30}>
          <div className="fork-head">
            <span className="fork-node fork-node--root mono">Demand · 15 min / 1 h / 24 h</span>
          </div>

          <div className="fork-rails" aria-hidden="true">
            <span className="fork-rail" />
            <span className="fork-rail" />
            <span className="fork-rail" />
          </div>

          <div className="experts-grid">
            {EXPERTS.map((e, i) => (
              <article className="expert-card" key={e.key}>
                <div className="expert-card-head">
                  <span className="expert-index mono">{`0${i + 1}`}</span>
                  <span className={`chip chip--${e.status === "live" ? "ok" : "accent"} chip-sm`}>
                    <span className="chip-dot" />
                    {e.status === "live" ? "live" : "in design"}
                  </span>
                </div>
                <h3 className="expert-name">{e.name}</h3>
                <p className="expert-family mono">{e.family}</p>
                <p className="expert-blurb">{e.blurb}</p>

                <dl className="expert-scores">
                  {e.h.map((v, hi) => (
                    <div className="expert-score" key={HORIS[hi]}>
                      <dt className="mono">{HORIS[hi]}</dt>
                      <dd className="mono">{v.toFixed(4)}</dd>
                    </div>
                  ))}
                </dl>
              </article>
            ))}
          </div>

          <div className="fork-rails fork-rails--out" aria-hidden="true">
            <span className="fork-rail" />
            <span className="fork-rail" />
            <span className="fork-rail" />
          </div>

          <div className="fork-head">
            <span className="fork-node fork-node--out mono">
              Fixed weighted ensemble · ships · {ENSEMBLE.h[2].toFixed(4)} kW @ 24 h
            </span>
          </div>
        </Reveal>

        <div className="experts-notes">
          <Reveal className="expert-note" delay={0} as="article">
            <span className="expert-note-label mono">Measured disagreement</span>
            <p>
              The three experts part company on{" "}
              <strong>
                {EXPERT_FACTS.disagreementMin}–{EXPERT_FACTS.disagreementMax}%
              </strong>{" "}
              of rows. A prediction is one voice among several; the spread between
              them is itself a measurement.
            </p>
          </Reveal>
          <Reveal className="expert-note" delay={110} as="article">
            <span className="expert-note-label mono">What ships</span>
            <p>
              A learned router was tested against the fixed ensemble and lost at{" "}
              <strong>
                {3 - EXPERT_FACTS.routerBeatsEnsemble} of 3 horizons
              </strong>
              . So the fixed weighted ensemble ships. This page does not claim the
              router is better, because it is not.
            </p>
          </Reveal>
          <Reveal className="expert-note" delay={220} as="article">
            <span className="expert-note-label mono">Not claimed</span>
            <p>
              No model here is presented as optimal. Each figure is a recorded
              measurement on the sealed test split, and the best single expert at
              each horizon is different from the best at another.
            </p>
          </Reveal>
        </div>

        <Reveal delay={80}>
          <p className="experts-foot mono">
            kW MAE · sealed test split · 179,520 rows · lower is better · recorded
            model output, not a live feed
          </p>
        </Reveal>
      </div>
    </section>
  );
}
