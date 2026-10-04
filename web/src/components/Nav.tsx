import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Logo from "./Logo";
import "./Nav.css";

const LINKS = [
  { href: "/#system", label: "System" },
  { href: "/#product", label: "Product" },
  { href: "/#evidence", label: "Evidence" },
  { href: "/#architecture", label: "Architecture" },
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
            <span className="chip nav-status" title="Phases 1-8 of 20 complete">
              <span className="chip-dot" />
              Phase 8 / 20 · research
            </span>
            <Link to="/console" className="btn btn-primary btn-sm nav-cta">
              Enter console
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
          Enter console
        </Link>
      </div>
    </>
  );
}
