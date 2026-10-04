import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import WordReveal from "../../components/anim/WordReveal";
import { DATASET_FACTS } from "../../data/system";
import "./Problem.css";

const TENSIONS = [
  {
    id: "01",
    title: "Generation moves, weather does not negotiate",
    body: "Renewable output arrives on its own schedule. On this 2018 feeder, a single real PV array's error at one day ahead is a missing-weather-forecast problem — the data decides what is knowable, not the modeller.",
  },
  {
    id: "02",
    title: "Forecasts disagree, and somebody must choose",
    body: "Phase 7 measured it: three heterogeneous experts disagree on 59–75% of rows. A prediction is not an answer; it is one voice among several, and the disagreement itself is information.",
  },
  {
    id: "03",
    title: "Confidence that looks calm can be wrong",
    body: "The default ±1σ interval — the rule everyone reaches for — failed 11 of 12 coverage cells on the sealed test split, in opposite directions at the two ends of the horizon range.",
  },
  {
    id: "04",
    title: "Operational decisions have consequences",
    body: "Executing the wrong action costs more than a wrong number. A decision pipeline needs to attack, simulate and verify a proposal before it becomes an action — and that machinery must earn trust the same way the forecast did.",
  },
];

export default function Problem() {
  return (
    <section className="section problem" id="problem">
      <div className="container">
        <SectionHead
          index="01"
          kicker="The problem"
          title={
            <>
              A prediction is not a decision.
              <br />
              Energy systems punish the difference.
            </>
          }
        />
        <WordReveal
          className="problem-lede"
          text="Demand shifts, renewable generation is uncertain, models disagree, and every operational action carries consequences. GridSentinal exists to connect the forecast to the decision — with the uncertainty carried the whole way."
        />
        <div className="problem-grid">
          {TENSIONS.map((t, i) => (
            <Reveal className="problem-item" key={t.id} delay={i * 90} as="article">
              <span className="problem-index mono">{t.id}</span>
              <h3>{t.title}</h3>
              <p>{t.body}</p>
            </Reveal>
          ))}
        </div>
        <Reveal delay={120}>
          <p className="problem-foot mono">
            {DATASET_FACTS.dataset} · {DATASET_FACTS.region} · feeder{" "}
            {DATASET_FACTS.feeder} · {DATASET_FACTS.points} points @{" "}
            {DATASET_FACTS.interval}
          </p>
        </Reveal>
      </div>
    </section>
  );
}
