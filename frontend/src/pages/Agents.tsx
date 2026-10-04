import { ScheduledPage } from "@/components/common";

export default function Agents() {
  return (
    <ScheduledPage
      week={3}
      title="Hospital Agents"
      planned="Four hospital cards with dataset size, model version, training status, live metrics and per-agent controls (train / pause / sync)."
    />
  );
}