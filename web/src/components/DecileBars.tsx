import { useMemo } from "react";

/**
 * Horizontal bars of realised error per interval-width decile.
 * Widths are sqrt-scaled because the ratio spans two orders of magnitude;
 * every bar carries its printed value, so the scaling cannot mislead.
 */
export default function DecileBars({
  deciles,
}: {
  deciles: number[];
}) {
  const max = useMemo(() => Math.max(...deciles), [deciles]);
  return (
    <div className="deciles" role="img"
      aria-label={`Realised error by interval-width decile, narrowest to widest: ${deciles
        .map((d) => `${d} times the mean error`)
        .join(", ")}.`}>
      {deciles.map((v, i) => {
        const last = i === deciles.length - 1;
        return (
          <div className="decile-row" key={i}>
            <span className="decile-label">D{i + 1}</span>
            <div className="decile-track">
              <div
                className={`decile-fill ${last ? "decile-fill--top" : ""}`}
                style={{ width: `${Math.max(2, (Math.sqrt(v) / Math.sqrt(max)) * 100)}%` }}
              />
            </div>
            <span className="decile-value">{v.toFixed(2)}×</span>
          </div>
        );
      })}
    </div>
  );
}
