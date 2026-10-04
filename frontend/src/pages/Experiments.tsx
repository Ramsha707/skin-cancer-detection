import { ScheduledPage } from "@/components/common";

export default function Experiments() {
  return (
    <ScheduledPage
      week={6}
      title="Fine-tuning & Experiments"
      planned="Frozen backbone vs selective fine-tuning vs full fine-tuning, compared on accuracy, precision, recall, specificity, F1, AUC, trainable parameters and wall-clock cost."
    />
  );
}