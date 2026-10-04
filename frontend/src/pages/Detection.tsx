import { ScheduledPage } from "@/components/common";

export default function Detection() {
  return (
    <ScheduledPage
      week={4}
      title="Cancer Detection"
      planned="Image upload, preprocessing, real MedSigLIP-448 inference across four classes, confidence breakdown, model version and inference latency."
    />
  );
}