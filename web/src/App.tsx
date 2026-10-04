import { useEffect } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import Landing from "./pages/Landing";
import Console from "./pages/Console";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  return (
    <>
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/console" element={<Console />} />
        <Route path="*" element={<Landing />} />
      </Routes>
    </>
  );
}
