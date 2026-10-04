import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ElementType,
  type ReactNode,
} from "react";
import "./anim.css";

/**
 * Scroll-entrance wrapper. Children start hidden (shifted + transparent)
 * and settle into place the first time they enter the viewport, with an
 * optional per-element delay for staggered groups.
 */
export default function Reveal({
  children,
  delay = 0,
  as: Tag = "div",
  className,
  rise = 26,
  role,
}: {
  children: ReactNode;
  delay?: number;
  as?: ElementType;
  className?: string;
  rise?: number;
  role?: string;
}) {
  const ref = useRef<HTMLElement | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[entries.length - 1]?.isIntersecting) {
          setInView(true);
          io.disconnect();
        }
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <Tag
      ref={ref}
      data-reveal=""
      role={role}
      className={`${inView ? "is-in" : ""} ${className ?? ""}`}
      style={{ "--reveal-delay": `${delay}ms`, "--reveal-rise": `${rise}px` } as CSSProperties}
    >
      {children}
    </Tag>
  );
}
