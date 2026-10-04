import { Link } from "react-router-dom";
import Logo from "./Logo";
import "./Footer.css";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div className="footer-brand">
            <div className="footer-brand-row">
              <Logo size={20} />
              <span>GridSentinal</span>
            </div>
            <p className="footer-tagline">
              An adaptive, self-verifying energy-management research system.
              Built for the Yuva Yodha Energy Tech Hackathon 2026, Schneider
              Electric.
            </p>
          </div>

          <nav className="footer-col" aria-label="Site">
            <h3>Site</h3>
            <a href="/#system">The system</a>
            <a href="/#product">Product preview</a>
            <a href="/#evidence">Evidence</a>
            <a href="/#architecture">Architecture</a>
          </nav>

          <nav className="footer-col" aria-label="Repository">
            <h3>Repository</h3>
            <span>README.md — measured results</span>
            <span>docs/architecture.md — component boundaries</span>
            <span>decisions.md — D-001 … D-109</span>
            <span>experiments/registry.jsonl — 29 records</span>
          </nav>

          <div className="footer-col">
            <h3>Status</h3>
            <span>Phases 1–8 of 20 complete</span>
            <span>Console: system status only</span>
            <Link to="/console">Enter console →</Link>
          </div>
        </div>

        <div className="footer-bottom">
          <span>
            Research system. No production deployment, no customers, no uptime
            claims — the evidence is the repository.
          </span>
          <span className="footer-license">
            Proprietary. All rights reserved.
          </span>
        </div>
      </div>
    </footer>
  );
}
