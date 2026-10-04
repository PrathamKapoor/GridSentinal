import SectionHead from "../../components/SectionHead";
import LoopDiagram from "../../components/LoopDiagram";

export default function LoopSection() {
  return (
    <section className="section" id="system">
      <div className="container">
        <SectionHead
          index="02"
          kicker="The intelligence loop"
          title="One loop, from telemetry to consequence."
          lede="GridSentinal is not a model that emits predictions. It is a closed loop that converts predictions into verified, risk-aware decisions — and then learns from the outcome. Select a stage to see what runs today."
        />
        <LoopDiagram />
      </div>
    </section>
  );
}
