import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import "./Verify.css";

const STEPS = [
  {
    label: "Propose",
    phase: "Phase 10 · optimizer",
    body: "Convert forecasts, uncertainty, flexibility and constraints into a candidate action.",
    status: "specified",
  },
  {
    label: "Simulate",
    phase: "Phase 11 · digital twin",
    body: "Run the proposal through an independent twin; compare its outcome with the learned model's prediction.",
    status: "specified",
  },
  {
    label: "Challenge",
    phase: "Phase 12 · red team",
    body: "Attack the decision: under what conditions does it become unsafe, ineffective or suboptimal?",
    status: "specified",
  },
  {
    label: "Verify",
    phase: "Phase 13 · assurance",
    body: "A gate between proposal and execution — approve or reject, with adaptive autonomy.",
    status: "specified",
  },
  {
    label: "Act",
    phase: "Phase 16 · execution",
    body: "Execute, observe the result, and return the outcome to the system as performance data.",
    status: "specified",
  },
];

const DISCIPLINE = [
  {
    title: "The sealed test split is read once",
    body: "Every phase evaluates on the same 179,520 chronological rows, read at the final evaluation only. No gate, policy or blend was added after seeing a test result — which is why the shipped configurations are not the best the ablations found.",
  },
  {
    title: "Success bars are registered before experiments run",
    body: "The Qwen phase's bar (beat 1.5648 kW at 24 h) was pre-registered in docs/qwen_energy_requirements.md before the phase ran. The model missed; the miss is published.",
  },
  {
    title: "Negative results are verdicts, not embarrassments",
    body: "Phase 6 (Qwen specialization) and Phase 7 (learned router) both returned negative results that were recorded rather than engineered around. Phase 7's router verdict remains negative in the shipped registry.",
  },
  {
    title: "The point forecast is pinned",
    body: "Phase 8 refuses to run unless the ensemble weights match Phase 7's recorded artifact to 0.0000% — every uncertainty number describes exactly the forecast that was published.",
  },
];

export default function Verify() {
  return (
    <section className="section verify" id="verify">
      <div className="container">
        <SectionHead
          index="05"
          kicker="Verify before act"
          title="An action is not executed because it was proposed."
          lede="The loop's defining discipline: a candidate action must survive simulation, red-teaming and an assurance gate before it moves anything in the physical system. The machinery is specified — the culture behind it is already running."
        />

        <ol className="verify-steps" aria-label="Decision verification sequence">
          {STEPS.map((s, i) => (
            <Reveal as="li" className="verify-step" key={s.label} delay={i * 100}>
              <div className="verify-step-node">
                <span className="verify-step-index mono">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="verify-step-label">{s.label}</span>
              </div>
              <p className="verify-step-phase mono">{s.phase}</p>
              <p className="verify-step-body">{s.body}</p>
              {i < STEPS.length - 1 && (
                <span className="verify-step-arrow" aria-hidden="true">
                  <svg width="16" height="10" viewBox="0 0 16 10" fill="none">
                    <path
                      d="M1 5h12.4M9.6 1.2 13.4 5 9.6 8.8"
                      stroke="currentColor"
                      strokeWidth="1.2"
                    />
                  </svg>
                </span>
              )}
            </Reveal>
          ))}
        </ol>

        <Reveal delay={80}>
        <div className="verify-discipline">
          <h3 className="verify-discipline-title">
            What already runs — the discipline, measured
          </h3>
          <div className="verify-discipline-grid">
            {DISCIPLINE.map((d) => (
              <article className="verify-discipline-item" key={d.title}>
                <h4>{d.title}</h4>
                <p>{d.body}</p>
              </article>
            ))}
          </div>
        </div>
        </Reveal>
      </div>
    </section>
  );
}
