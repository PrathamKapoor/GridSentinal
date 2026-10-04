import type { ReactNode } from "react";

export default function SectionHead({
  index,
  kicker,
  title,
  lede,
  children,
}: {
  index: string;
  kicker: string;
  title: ReactNode;
  lede?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="section-head">
      <p className="kicker">
        {index} · {kicker}
      </p>
      <h2 className="section-heading">{title}</h2>
      {lede && <p className="section-lede">{lede}</p>}
      {children}
    </div>
  );
}
