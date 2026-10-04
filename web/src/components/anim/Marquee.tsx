import type { ReactNode } from "react";
import "./anim.css";

/**
 * Infinite horizontal ticker. The track is rendered twice so the loop is
 * seamless; the animation is pure CSS and disabled under reduced motion.
 */
export default function Marquee({
  items,
  speed = 36,
  className = "",
  separator,
  ariaLabel,
}: {
  items: ReactNode[];
  speed?: number;
  className?: string;
  separator?: ReactNode;
  ariaLabel?: string;
}) {
  const sep = separator ?? (
    <span className="marquee-sep" aria-hidden="true">
      ◆
    </span>
  );
  const group = (hidden: boolean) => (
    <div className="marquee-group" aria-hidden={hidden || undefined}>
      {items.map((item, i) => (
        <span className="marquee-item" key={i}>
          {item}
          {sep}
        </span>
      ))}
    </div>
  );

  return (
    <div className={`marquee ${className}`} style={{ "--marquee-speed": `${speed}s` } as React.CSSProperties}>
      <div className="marquee-track">
        {group(false)}
        {group(true)}
      </div>
      <p className="sr-only">{ariaLabel ?? items.map((i) => String(i)).join(", ")}</p>
    </div>
  );
}
