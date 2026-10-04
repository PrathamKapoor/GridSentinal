import { useEffect, useRef, useState } from "react";

/**
 * Counts up to a value the first time it scrolls into view.
 * Honors prefers-reduced-motion by rendering the final value immediately.
 */
export default function CountUp({
  value,
  duration = 1300,
  format = (v: number) => Math.round(v).toLocaleString("en-US"),
}: {
  value: number;
  duration?: number;
  format?: (v: number) => string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [display, setDisplay] = useState(() =>
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? format(value)
      : format(0),
  );

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let raf = 0;
    let started = false;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries[entries.length - 1]?.isIntersecting || started) return;
        started = true;
        io.disconnect();
        const start = performance.now();
        const tick = (now: number) => {
          const t = Math.min(1, (now - start) / duration);
          const eased = 1 - Math.pow(1 - t, 3);
          setDisplay(format(value * eased));
          if (t < 1) raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
      },
      { threshold: 0.4 },
    );
    io.observe(el);
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [value, duration, format]);

  return (
    <span ref={ref} className="countup">
      {display}
    </span>
  );
}
