/**
 * GridSentinal mark: a sentinel orbit. One ring carries a signal node;
 * an energy pulse crosses the centre. Reads at 16px, scales cleanly.
 */
export default function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <defs>
        <linearGradient id="gs-pulse" x1="4" y1="16" x2="20" y2="8" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#2f9e63" />
          <stop offset="0.55" stopColor="#4ade80" />
          <stop offset="1" stopColor="#c9fbdd" />
        </linearGradient>
      </defs>

      {/* orbit ring, open where the sentinel sits */}
      <path
        d="M17.83 6.17 A 8.25 8.25 0 1 0 20.25 12"
        stroke="currentColor"
        strokeOpacity="0.5"
        strokeWidth="1.5"
        strokeLinecap="round"
      />

      {/* energy pulse */}
      <path
        d="M4.4 12h3.2l1.65-3.8 2.5 7.6 1.65-3.8h4.1"
        stroke="url(#gs-pulse)"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* sentinel node on the ring */}
      <circle cx="17.83" cy="6.17" r="1.9" fill="#4ade80" />
      <circle cx="17.83" cy="6.17" r="3.1" stroke="#4ade80" strokeOpacity="0.35" />
    </svg>
  );
}
