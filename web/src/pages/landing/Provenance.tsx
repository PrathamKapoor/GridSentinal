import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import { PROVENANCE, PROVENANCE_FOOTNOTE } from "../../data/landing";
import "./Provenance.css";

export default function Provenance() {
  return (
    <section className="section provenance" id="provenance">
      <div className="container">
        <SectionHead
          index="09"
          kicker="Provenance + integrity"
          title="Every number can be walked back to its source."
          lede="Checksums, protocol freezes, configuration hashes and per-cell artifacts are not ceremony. They are the reason a figure in a report can be traced to the rows that produced it, and the reason a rule cannot be quietly rewritten after a run."
        />

        <ol className="chain" aria-label="Provenance chain">
          {PROVENANCE.map((s, i) => (
            <Reveal
              as="li"
              className="chain-step"
              key={s.key}
              delay={Math.min(i * 100, 400)}
            >
              <span className="chain-index mono" aria-hidden="true">
                {String(i + 1).padStart(2, "0")}
              </span>

              <div className="chain-card">
                <span className="chain-label mono">{s.label}</span>
                <p className="chain-detail">{s.detail}</p>
                <div className="chain-evidence">
                  <span className="chain-hash mono">{s.evidence}</span>
                  <span className="chain-evidence-title mono">{s.evidenceTitle}</span>
                </div>
              </div>

              {i < PROVENANCE.length - 1 && (
                <span className="chain-arrow" aria-hidden="true">
                  <svg width="18" height="10" viewBox="0 0 18 10" fill="none">
                    <path
                      d="M1 5h13.6M11 1.2 15 5l-4 3.8"
                      stroke="currentColor"
                      strokeWidth="1.2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </span>
              )}
            </Reveal>
          ))}
        </ol>

        <Reveal delay={80}>
          <div className="chain-integrity">
            <div className="chain-integrity-item">
              <span className="chain-integrity-label mono">Final test split</span>
              <span className="chain-integrity-value mono">locked · read once</span>
            </div>
            <div className="chain-integrity-item">
              <span className="chain-integrity-label mono">Feature selection</span>
              <span className="chain-integrity-value mono">restricted in code</span>
            </div>
            <div className="chain-integrity-item">
              <span className="chain-integrity-label mono">Binaries committed</span>
              <span className="chain-integrity-value mono">0</span>
            </div>
            <div className="chain-integrity-item">
              <span className="chain-integrity-label mono">Dataset</span>
              <span className="chain-integrity-value mono">SMART-DS · public</span>
            </div>
          </div>
        </Reveal>

        <Reveal delay={140}>
          <p className="chain-foot mono">{PROVENANCE_FOOTNOTE}</p>
        </Reveal>
      </div>
    </section>
  );
}
