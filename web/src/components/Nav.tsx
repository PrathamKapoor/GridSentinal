import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Logo from "./Logo";
import "./Nav.css";

/**
 * Landing anchors. Every entry points at a section that exists on the page;
 * there are no placeholder or dead links. `simulation` resolves to the
 * architecture section, which is where the digital-twin layer and its real
 * (specified, not implemented) status are documented.
 */
const LINKS = [
  { href: "/#system", label: "System" },
  { href: "/#forecast", label: "Forecast" },
  { href: "/#flexibility", label: "Flexibility" },
  { href: "/#decisions", label: "Decisions" },
  { href: "/#architecture", label: "Simulation" },
  { href: "/#research", label: "Research" },
];

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // close the mobile menu when leaving the page
  useEffect(() => () => setOpen(false), []);

  return (
    <>
      <header className={`nav ${scrolled || open ? "nav--solid" : ""}`}>
        <div className="nav-shell">
          <Link to="/" className="nav-brand" aria-label="GridSentinal home">
            <Logo />
            <span className="nav-wordmark">
              GridSentinal
              <span className="nav-wordmark-suffix">research</span>
            </span>
          </Link>

          <nav className="nav-links" aria-label="Site">
            {LINKS.map((l) => (
              <a key={l.href} href={l.href} className="nav-link">
                {l.label}
              </a>
            ))}
          </nav>

          <div className="nav-right">
            <span className="chip nav-status" title="Research system, pre-release">
              <span className="chip-dot" />
              research system
            </span>
            <Link to="/console" className="btn btn-primary btn-sm nav-cta">
              Open console
            </Link>
            <button
              className="nav-burger"
              aria-expanded={open}
              aria-controls="mobile-menu"
              aria-label={open ? "Close menu" : "Open menu"}
              onClick={() => setOpen((v) => !v)}
            >
              <span className={`nav-burger-bar ${open ? "is-open" : ""}`} />
              <span className={`nav-burger-bar ${open ? "is-open" : ""}`} />
            </button>
          </div>
        </div>
      </header>

      <div
        id="mobile-menu"
        className={`nav-mobile ${open ? "is-open" : ""}`}
        hidden={!open}
      >
        {LINKS.map((l) => (
          <a
            key={l.href}
            href={l.href}
            className="nav-mobile-link"
            onClick={() => setOpen(false)}
          >
            {l.label}
          </a>
        ))}
        <Link
          to="/console"
          className="btn btn-primary nav-mobile-cta"
          onClick={() => setOpen(false)}
        >
          Open console
        </Link>
      </div>
    </>
  );
}
