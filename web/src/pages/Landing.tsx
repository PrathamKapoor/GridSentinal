import Nav from "../components/Nav";
import Footer from "../components/Footer";
import Hero from "./landing/Hero";
import Problem from "./landing/Problem";
import LoopSection from "./landing/LoopSection";
import Experts from "./landing/Experts";
import Uncertainty from "./landing/Uncertainty";
import Flexibility from "./landing/Flexibility";
import Verify from "./landing/Verify";
import Research from "./landing/Research";
import Evidence from "./landing/Evidence";
import Provenance from "./landing/Provenance";
import ProductPreview from "./landing/ProductPreview";
import Architecture from "./landing/Architecture";
import FinalCta from "./landing/FinalCta";

/**
 * The landing narrative, in order:
 *   problem -> loop -> experts -> uncertainty -> flexibility ->
 *   self-verification -> research -> evidence -> provenance ->
 *   command center preview -> architecture -> enter
 */
export default function Landing() {
  return (
    <>
      <Nav />
      <main id="main">
        <Hero />
        <Problem />
        <LoopSection />
        <Experts />
        <Uncertainty />
        <Flexibility />
        <Verify />
        <Research />
        <Evidence />
        <Provenance />
        <ProductPreview />
        <Architecture />
        <FinalCta />
      </main>
      <Footer />
    </>
  );
}
