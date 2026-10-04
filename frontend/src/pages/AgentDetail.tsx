import { ScheduledPage } from "@/components/common";

export default function AgentDetail() {
  return (
    <ScheduledPage
      week={3}
      title="Hospital Detail"
      planned="Local dataset statistics, class distribution, per-epoch local training curves, model version history, privacy status and federated participation record."
    />
  );
}