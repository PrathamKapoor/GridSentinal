import { useEffect, useRef } from "react";
import "./anim.css";

/**
 * Scroll-driven word reveal: words start dim and light up one after
 * another as the block travels through the viewport. Pure DOM + one
 * rAF-throttled scroll listener; no dependencies.
 */
export default function WordReveal({
  text,
  className = "",
}: {
  text: string;
  className?: string;
}) {
  const ref = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.classList.add("is-lit");
      return;
    }

    const words = Array.from(el.querySelectorAll<HTMLElement>(".wr-word"));
    let raf = 0;
    let ticking = false;

    const update = () => {
      ticking = false;
      const rect = el.getBoundingClientRect();
      const vh = window.innerHeight;
      // progress 0 → 1 as the block moves from 88% to 38% of the viewport
      const p = (vh * 0.88 - rect.top) / (vh * 0.5);
      const clamped = Math.max(0, Math.min(1, p));
      const lit = clamped * words.length;
      for (let i = 0; i < words.length; i++) {
        words[i]!.classList.toggle("is-lit", i < lit);
      }
    };

    const onScroll = () => {
      if (!ticking) {
        ticking = true;
        raf = requestAnimationFrame(update);
      }
    };

    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [text]);

  return (
    <p ref={ref} className={`wr ${className}`}>
      {text.split(" ").map((word, i) => (
        <span className="wr-word" key={i}>
          {word}{" "}
        </span>
      ))}
    </p>
  );
}
