import { ScheduledPage } from "@/components/common";

export default function Performance() {
  return (
    <ScheduledPage
      week={9}
      title="Model Performance & Evaluation"
      planned="Accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, confusion matrix, plus per-round learning curves."
    />
  );
}