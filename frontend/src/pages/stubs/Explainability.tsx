import { ScheduledPage } from "@/components/common";

export default function Explainability() {
  return (
    <ScheduledPage
      week={10}
      title="Explainability (Grad-CAM)"
      planned="Gradient-weighted Class Activation Mapping over the fine-tuned SigLIP vision tower, rendered as a heatmap overlay on the uploaded lesion."
    />
  );
}