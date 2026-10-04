import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import { PHASE_LEDGER } from "../../data/landing";
import "./Research.css";

const VERDICT_TONE: Record<string, string> = {
  COMPLETE: "ok",
  YES: "ok",
  NEGATIVE: "fail",
  NO: "fail",
  "SEE REPORT": "warn",
};

export default function Research() {
  return (
    <section className="section research" id="research">
      <div className="container">
        <SectionHead
          index="07"
          kicker="Research, not magic"
          title={
            <>
              Built to fail
              <br />
              <em>honestly.</em>
            </>
          }
          lede="Every experiment leaves evidence, and every negative result stays visible. A research system that only publishes its wins is not measuring anything — it is marketing."
        />

        <ol className="ledger" aria-label="Research phase ledger">
          {PHASE_LEDGER.map((p, i) => (
            <Reveal
              as="li"
              className="ledger-row"
              key={p.n}
              delay={Math.min(i * 90, 420)}
            >
              <div className="ledger-rail" aria-hidden="true">
                <span className="ledger-node" />
                {i < PHASE_LEDGER.length - 1 && <span className="ledger-line" />}
              </div>

              <div className="ledger-body">
                <div className="ledger-head">
                  <span className="ledger-phase mono">
                    Phase {String(p.n).padStart(2, "0")}
                  </span>
                  <span className={`chip chip--${VERDICT_TONE[p.verdict] ?? "accent"}`}>
                    <span className="chip-dot" />
                    {p.verdict}
                  </span>
                </div>

                <h3 className="ledger-name">{p.name}</h3>
                <p className="ledger-headline mono">{p.headline}</p>
                <p className="ledger-note">{p.note}</p>
              </div>
            </Reveal>
          ))}
        </ol>

        <Reveal delay={120}>
          <div className="research-principle">
            <p className="research-principle-title">
              A negative result is a verdict, not an embarrassment.
            </p>
            <p className="research-principle-body">
              Success bars are registered before experiments run, the sealed test
              split is read once, and a configuration that beats the shipped one is
              still not swapped in afterwards — because doing so would be tuning
              against data that was meant to be unseen. The result is a system whose
              surviving numbers mean something.
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
