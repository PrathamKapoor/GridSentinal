import { useState } from "react";
import { LOOP, type LoopStage } from "../data/system";
import Reveal from "./anim/Reveal";
import "./LoopDiagram.css";

const OPERATIONAL_COUNT = LOOP.filter((s) => s.status === "operational").length;

/**
 * The intelligence loop: eleven stages on a wrapped rail, each focusable,
 * with a detail panel below. Implemented stages are marked operational;
 * the rest are honestly labelled specified, with the layer that owns them.
 */
export default function LoopDiagram() {
  const [active, setActive] = useState<LoopStage>(
    LOOP.find((s) => s.key === "predict") ?? LOOP[0]!,
  );

  return (
    <div className="loop">
      <Reveal>
        <ol className="loop-rail" aria-label="Intelligence loop stages">
          {LOOP.map((stage, i) => {
            const isActive = stage.key === active.key;
            const isOperational = stage.status === "operational";
            return (
              <li className="loop-node-wrap" key={stage.key}>
                <button
                  className={`loop-node ${isActive ? "is-active" : ""} ${
                    isOperational ? "is-operational" : "is-specified"
                  }`}
                  onClick={() => setActive(stage)}
                  aria-pressed={isActive}
                >
                  <span className="loop-node-index">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span className="loop-node-label">{stage.label}</span>
                  <span
                    className={`loop-node-status ${
                      isOperational ? "is-operational" : ""
                    }`}
                  >
                    {isOperational ? "operational" : "specified"}
                  </span>
                </button>
                {i < LOOP.length - 1 && (
                  <span
                    className={`loop-arrow ${isOperational ? "is-operational" : ""}`}
                    aria-hidden="true"
                  >
                    <svg width="14" height="10" viewBox="0 0 14 10" fill="none">
                      <path
                        d="M1 5h10.4M8.6 1.4 12.2 5 8.6 8.6"
                        stroke="currentColor"
                        strokeWidth="1.3"
                      />
                    </svg>
                  </span>
                )}
                {/* loop-back connector on the last node */}
                {i === LOOP.length - 1 && (
                  <span
                    className="loop-arrow loop-arrow--back"
                    aria-hidden="true"
                    title="back to observe"
                  >
                    <svg width="14" height="10" viewBox="0 0 14 10" fill="none">
                      <path
                        d="M13 5H2.6M5.4 1.4 1.8 5l3.6 3.6"
                        stroke="currentColor"
                        strokeWidth="1.3"
                      />
                    </svg>
                  </span>
                )}
              </li>
            );
          })}
        </ol>
      </Reveal>

      <Reveal delay={140}>
        <div className="loop-detail panel" aria-live="polite">
          <div className="loop-detail-head">
            <span className="loop-detail-step mono">
              {String(LOOP.indexOf(active) + 1).padStart(2, "0")} / {LOOP.length}
            </span>
            <h3 className="loop-detail-title">{active.label}</h3>
            <span
              className={`chip ${
                active.status === "operational" ? "chip--ok" : "chip--accent"
              }`}
            >
              <span className="chip-dot" />
              {active.status === "operational"
                ? `operational · ${active.owner}`
                : `specified · ${active.owner}`}
            </span>
          </div>
          <p className="loop-detail-body">{active.description}</p>
        </div>
      </Reveal>

      <Reveal delay={220}>
        <p className="loop-note source-note">
          {OPERATIONAL_COUNT} of {LOOP.length} stages run today, end to end, on a
          real 2018 feeder dataset. The remaining stages are specified in
          docs/architecture.md; the loop is the plan the implementation is held to,
          not a rendering of finished software.
        </p>
      </Reveal>
    </div>
  );
}
