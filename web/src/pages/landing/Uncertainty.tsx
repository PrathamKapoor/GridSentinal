import SectionHead from "../../components/SectionHead";
import Reveal from "../../components/anim/Reveal";
import DecileBars from "../../components/DecileBars";
import evidence from "../../data/evidence.json";
import "./Uncertainty.css";

interface Procedure {
  horizon: number;
  method: string;
  coverage_90: number;
  coverage_error: number;
  spearman_width_error: number;
  top_decile_multiple: number;
  deciles: number[];
}

interface FailureCell {
  horizon: number;
  nominal_level: number;
  coverage: number;
  passes: boolean;
}

const procedures = evidence.published_procedure as Procedure[];
const failure = evidence.constant_width_failure as {
  cells: FailureCell[];
  failing_cells: number;
  total_cells: number;
};
const h1 = procedures[0]!;

const LEVELS = [0.5, 0.8, 0.9, 0.95];
const HORIZONS = [1, 4, 96];

function cellFor(level: number, horizon: number): FailureCell | undefined {
  return failure.cells.find(
    (c) => Math.abs(c.nominal_level - level) < 1e-9 && c.horizon === horizon,
  );
}

export default function Uncertainty() {
  return (
    <section className="section uncertainty" id="uncertainty">
      <div className="container">
        <SectionHead
          index="04"
          kicker="Uncertainty"
          title="Every forecast carries its own doubt — and says how much."
          lede="GridSentinal does not stop at a point prediction. Six interval methods were fitted, calibrated and scored once on a sealed test split; the published procedure reports its own coverage, and the widths move with the difficulty of the day."
        />

        <div className="uncertainty-grid">
          <Reveal className="uncertainty-reveal">
          <div className="panel uncertainty-table-panel">
            <div className="panel-head">
              <span className="panel-title">
                Published procedure · sealed test split
              </span>
            </div>
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Horizon</th>
                    <th scope="col">Method</th>
                    <th scope="col" className="num">
                      Cov @90%
                    </th>
                    <th scope="col" className="num">
                      ρ width↔err
                    </th>
                    <th scope="col" className="num">
                      Top decile
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {procedures.map((p) => (
                    <tr key={p.horizon}>
                      <td className="mono">h={p.horizon}</td>
                      <td className="mono">{p.method}</td>
                      <td className="mono num ok-text">
                        {(p.coverage_90 * 100).toFixed(1)}%
                      </td>
                      <td className="mono num">+{p.spearman_width_error.toFixed(3)}</td>
                      <td className="mono num">{p.top_decile_multiple.toFixed(2)}×</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="table-note">
              Coverage within ±0.01 of nominal at every horizon. The widest
              tenth of the intervals carries {h1.top_decile_multiple.toFixed(1)}×
              the mean error — the system knows which rows to distrust.
            </p>

            <div className="uncertainty-deciles">
              <p className="panel-title panel-title--gap">
                Realised error by interval width · h=1 · narrowest → widest
              </p>
              <DecileBars deciles={h1.deciles} />
            </div>
          </div>
          </Reveal>

          <Reveal delay={140} className="uncertainty-reveal">
          <div className="panel uncertainty-failure">
            <div className="panel-head">
              <span className="panel-title">The negative finding</span>
              <span className="chip chip--fail">
                <span className="chip-dot" />
                {failure.failing_cells} / {failure.total_cells} cells fail
              </span>
            </div>
            <p className="uncertainty-failure-lede">
              The constant-width <span className="mono">±1σ</span> interval —
              the rule a practitioner reaches for without thinking — is not
              calibrated on this data:
            </p>
            <div className="table-scroll">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Nominal</th>
                    {HORIZONS.map((h) => (
                      <th scope="col" className="num" key={h}>
                        h={h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {LEVELS.map((level) => (
                    <tr key={level}>
                      <td className="mono">{(level * 100).toFixed(0)}%</td>
                      {HORIZONS.map((h) => {
                        const cell = cellFor(level, h);
                        if (!cell) return <td key={h} className="mono num">—</td>;
                        return (
                          <td
                            key={h}
                            className={`mono num ${
                              cell.passes ? "ok-text" : "fail-text"
                            }`}
                          >
                            {(cell.coverage * 100).toFixed(1)}
                            {cell.passes ? " ✓" : " ✕"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="table-note">
              Constant-width <span className="mono">global_residual</span>{" "}
              coverage, sealed test split. It over-covers at 15 minutes and
              under-covers at 24 hours, in opposite directions — demand
              difficulty moves across the year, and a width calibrated on one
              half does not transfer to the other.
            </p>
            <div className="uncertainty-honesty">
              <p>
                <strong>Not claimed:</strong> a coverage guarantee. Split
                conformal's guarantee is conditional on exchangeability, and the
                phase's own split-parity numbers show that assumption is
                violated. Every interval carries that caveat in its metadata;
                time-ordered conformal is the recorded direction for earning a
                real one.
              </p>
            </div>
          </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
