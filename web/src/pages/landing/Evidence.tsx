import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import CountUp from "../../components/anim/CountUp";
import evidence from "../../data/evidence.json";
import "./Evidence.css";

interface Phase7Row {
  method: string;
  h1: number;
  h4: number;
  h96: number;
}

const rows = evidence.phase7_table as Phase7Row[];
const facts = evidence.router_facts as {
  disagreement_pct_min: number;
  disagreement_pct_max: number;
  oracle_gain_pct_min: number;
  oracle_gain_pct_max: number;
  horizons_router_beats_fixed_ensemble: number;
};

const NEGATIVES = [
  {
    id: "Phase 6",
    title: "Language-model specialization",
    verdict: "Negative — recorded",
    body: "A frozen Qwen3-1.7B trunk, adapted with a task head, lost to the classical GBM at every horizon: 2.74 vs 1.56 kW MAE at 24 hours, on identical rows. Pretraining transfers (33.9% better than a random trunk at 15 min) but the readout is nearly rank-2.",
  },
  {
    id: "Phase 7",
    title: "Learned expert router",
    verdict: `Negative — ${facts.horizons_router_beats_fixed_ensemble} of 3 horizons`,
    body: "The diversity premise held — experts disagree on 59–75% of rows and 30–38% of achievable gain exists — but the learned gate captured −2% to 13% of it and lost to a fixed weighted ensemble everywhere.",
  },
  {
    id: "Phase 5",
    title: "The shipped config is not the best found",
    verdict: "Discipline over metrics",
    body: "A no-cyclic-channels ablation beat the shipped configuration at all three horizons. The channels stayed: dropping them after seeing the test split would be tuning against sealed data (D-077).",
  },
];

export default function Evidence() {
  return (
    <section className="section" id="evidence">
      <div className="container">
        <SectionHead
          index="06"
          kicker="Evidence"
          title="Every claim has a receipt."
          lede="Models are evaluated, challenged, and retained only when evidence supports them. The project is comfortable publishing what failed — the failures are the reason the surviving numbers mean something."
        />

        <Reveal>
        <div className="evidence-table panel">
          <div className="panel-head">
            <span className="panel-title">
              Demand forecast · kW MAE · sealed test split · 179,520 rows
            </span>
            <span className="chip">lower is better</span>
          </div>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Method</th>
                  <th scope="col" className="num">
                    h=1 · 15 min
                  </th>
                  <th scope="col" className="num">
                    h=4 · 1 h
                  </th>
                  <th scope="col" className="num">
                    h=96 · 24 h
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const highlight = r.method === "Fixed weighted ensemble";
                  const oracle = r.method === "Oracle expert assignment";
                  return (
                    <tr
                      key={r.method}
                      className={highlight ? "evidence-row-ship" : oracle ? "evidence-row-oracle" : ""}
                    >
                      <td>
                        {r.method}
                        {highlight && (
                          <span className="evidence-ship-tag mono">ships</span>
                        )}
                      </td>
                      <td className="mono num">{r.h1.toFixed(4)}</td>
                      <td className="mono num">{r.h4.toFixed(4)}</td>
                      <td className="mono num">{r.h96.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="table-note">
            The learned router lost to the fixed weighted ensemble at{" "}
            {facts.horizons_router_beats_fixed_ensemble} of 3 horizons — so the
            ensemble ships. The oracle row bounds what a perfect router would
            reach; the gap is the honest size of the remaining routing
            opportunity, not a capability claim.
          </p>
        </div>
        </Reveal>

        <div className="evidence-negatives">
          {NEGATIVES.map((n, i) => (
            <Reveal key={n.id} delay={i * 110} as="article" className="evidence-negative-reveal">
            <article className="panel evidence-negative">
              <div className="panel-head">
                <span className="panel-title">{n.id}</span>
                <span className="chip chip--warn">
                  <span className="chip-dot" />
                  {n.verdict}
                </span>
              </div>
                <div className="evidence-negative-body">
                  <h3>{n.title}</h3>
                  <p>{n.body}</p>
                </div>
            </article>
            </Reveal>
          ))}
        </div>

        <Reveal delay={100}>
        <div className="evidence-strip" role="list" aria-label="Process facts">
          <div className="evidence-strip-item" role="listitem">
            <span className="evidence-strip-value mono"><CountUp value={1235} /></span>
            <span className="evidence-strip-label">tests passing</span>
          </div>
          <div className="evidence-strip-item" role="listitem">
            <span className="evidence-strip-value mono"><CountUp value={29} /></span>
            <span className="evidence-strip-label">experiments registered</span>
          </div>
          <div className="evidence-strip-item" role="listitem">
            <span className="evidence-strip-value mono">D-109</span>
            <span className="evidence-strip-label">decision records</span>
          </div>
          <div className="evidence-strip-item" role="listitem">
            <span className="evidence-strip-value mono">0</span>
            <span className="evidence-strip-label">
              binaries or datasets committed
            </span>
          </div>
        </div>
        </Reveal>
      </div>
    </section>
  );
}
