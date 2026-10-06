import { ScheduledPage } from "@/components/common";

export default function ModelRegistry() {
  return (
    <ScheduledPage
      week={12}
      title="Model Registry"
      planned="Every global model version with its training strategy, originating federated round, held-out metrics, contributing agents and lifecycle status."
    />
  );
}