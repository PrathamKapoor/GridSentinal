import Nav from "../components/Nav";
import Footer from "../components/Footer";
import Hero from "./landing/Hero";
import Problem from "./landing/Problem";
import LoopSection from "./landing/LoopSection";
import ProductPreview from "./landing/ProductPreview";
import Uncertainty from "./landing/Uncertainty";
import Verify from "./landing/Verify";
import Evidence from "./landing/Evidence";
import Architecture from "./landing/Architecture";
import FinalCta from "./landing/FinalCta";

export default function Landing() {
  return (
    <>
      <Nav />
      <main id="main">
        <Hero />
        <Problem />
        <LoopSection />
        <ProductPreview />
        <Uncertainty />
        <Verify />
        <Evidence />
        <Architecture />
        <FinalCta />
      </main>
      <Footer />
    </>
  );
}
