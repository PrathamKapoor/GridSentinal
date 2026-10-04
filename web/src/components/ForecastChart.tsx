import { useMemo, useRef, useState } from "react";
import "./charts.css";

export interface ForecastSeries {
  timestamps: string[];
  point: number[];
  lower: number[];
  upper: number[];
}

interface Props {
  series: ForecastSeries;
  /** asset / method caption shown in the aria-label */
  caption: string;
  unit?: string;
  height?: number;
  /** show day-boundary separators (timestamps must be ISO) */
  daySeparators?: boolean;
}

const PAD = { top: 14, right: 12, bottom: 26, left: 44 };
const W = 760; // viewBox width; the svg scales responsively

function fmtTime(iso: string) {
  return iso.slice(11, 16);
}

/**
 * Engineering forecast chart: point line, calibrated interval band,
 * unit-labelled axes, restrained gridlines, day boundaries, and a
 * pointer crosshair with a readable tooltip.
 */
export default function ForecastChart({
  series,
  caption,
  unit = "kW",
  height = 260,
  daySeparators = true,
}: Props) {
  const H = height;
  const { point, lower, upper, timestamps } = series;
  const n = point.length;
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const geom = useMemo(() => {
    const lo = Math.min(...lower);
    const hi = Math.max(...upper);
    const padY = (hi - lo) * 0.08 || 0.5;
    const yMin = Math.max(0, lo - padY);
    const yMax = hi + padY;
    const iw = W - PAD.left - PAD.right;
    const ih = H - PAD.top - PAD.bottom;
    const x = (i: number) => PAD.left + (i / (n - 1)) * iw;
    const y = (v: number) => PAD.top + (1 - (v - yMin) / (yMax - yMin)) * ih;

    const at = (arr: number[], i: number) => arr[i] ?? 0;
    const band =
      `M ${lower.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" L ")}` +
      " L " +
      [...upper.keys()]
        .reverse()
        .map((i) => `${x(i).toFixed(1)},${y(at(upper, i)).toFixed(1)}`)
        .join(" L ") +
      " Z";
    const line = point
      .map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`)
      .join(" L ");

    // ~6 y ticks
    const ticks: number[] = [];
    for (let i = 0; i <= 4; i++) {
      ticks.push(yMin + ((yMax - yMin) * i) / 4);
    }
    // one x label every 24 steps (6 h), offset to avoid the edge
    const xTicks: number[] = [];
    const step = 24;
    for (let i = 0; i < n; i += step) xTicks.push(i);

    // day boundaries: local midnights in the UTC-stamped series
    const days: number[] = [];
    if (daySeparators) {
      let last = timestamps[0]?.slice(8, 10) ?? "";
      for (let i = 1; i < n; i++) {
        const d = timestamps[i]?.slice(8, 10) ?? "";
        if (d !== last) {
          days.push(i);
          last = d;
        }
      }
    }

    return { x, y, band, line, ticks, xTicks, days, yMin, yMax, iw, ih };
  }, [point, lower, upper, timestamps, n, H, daySeparators]);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.round(
      ((px - PAD.left) / geom.iw) * (n - 1),
    );
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  const hi = hover != null ? hover : null;
  const tipLeftPct =
    hi != null ? (geom.x(hi) / W) * 100 : 0;
  const tipRight = tipLeftPct > 78;

  return (
    <div className="chart">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${caption}. Demand in kilowatts over ${n} fifteen-minute steps, with the calibrated 90 percent interval band. Point forecast ranges from ${Math.min(...point).toFixed(2)} to ${Math.max(...point).toFixed(2)} kW.`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
      >
        {/* gridlines + y labels */}
        {geom.ticks.map((t, i) => (
          <g key={i}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={geom.y(t)}
              y2={geom.y(t)}
              className="chart-grid"
            />
            <text x={PAD.left - 8} y={geom.y(t) + 3} className="chart-tick" textAnchor="end">
              {t.toFixed(1)}
            </text>
          </g>
        ))}

        {/* day boundaries */}
        {geom.days.map((i) => (
          <g key={i}>
            <line
              x1={geom.x(i)}
              x2={geom.x(i)}
              y1={PAD.top}
              y2={H - PAD.bottom}
              className="chart-dayline"
            />
            <text x={geom.x(i) + 5} y={PAD.top + 11} className="chart-daylabel">
              {timestamps[i]!.slice(0, 10)}
            </text>
          </g>
        ))}

        {/* interval band + point line */}
        <path d={geom.band} className="chart-band" />
        <path d={`M ${geom.line}`} className="chart-line" pathLength={1} />

        {/* x labels */}
        {geom.xTicks.map((i) => (
          <text
            key={i}
            x={geom.x(i)}
            y={H - 8}
            className="chart-tick"
            textAnchor="middle"
          >
            {fmtTime(timestamps[i]!)}
          </text>
        ))}
        <text x={PAD.left - 8} y={8} className="chart-unit" textAnchor="end">
          {unit}
        </text>

        {/* crosshair */}
        {hi != null && (
          <g>
            <line
              x1={geom.x(hi)}
              x2={geom.x(hi)}
              y1={PAD.top}
              y2={H - PAD.bottom}
              className="chart-crosshair"
            />
            <circle
              cx={geom.x(hi)}
              cy={geom.y(point[hi]!)}
              r={3.2}
              className="chart-dot"
            />
            <circle
              cx={geom.x(hi)}
              cy={geom.y(upper[hi]!)}
              r={2}
              className="chart-dot chart-dot--dim"
            />
            <circle
              cx={geom.x(hi)}
              cy={geom.y(lower[hi]!)}
              r={2}
              className="chart-dot chart-dot--dim"
            />
          </g>
        )}
      </svg>

      {hi != null && timestamps[hi] && (
        <div
          className="chart-tooltip"
          role="status"
          style={{
            left: `${tipLeftPct}%`,
            transform: tipRight ? "translateX(-108%)" : "translateX(8px)",
          }}
        >
          <span className="chart-tooltip-time">{fmtTime(timestamps[hi]!)}</span>
          <span className="chart-tooltip-val">
            {point[hi]!.toFixed(2)} {unit}
          </span>
          <span className="chart-tooltip-band">
            90% band {lower[hi]!.toFixed(2)} – {upper[hi]!.toFixed(2)}{" "}
            {(upper[hi]! - lower[hi]!).toFixed(2)} {unit} wide
          </span>
        </div>
      )}
    </div>
  );
}
