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
      {/* the loop: observe -> predict -> verify -> act, drawn as one circuit */}
      <rect
        x="3.5"
        y="3.5"
        width="17"
        height="17"
        rx="1.5"
        stroke="currentColor"
        strokeOpacity="0.35"
      />
      <path
        d="M7.5 15.5V8.5H12a2.5 2.5 0 0 1 0 5H8.2"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="square"
      />
      <circle cx="16.5" cy="8" r="1.3" fill="#4ade80" />
    </svg>
  );
}
