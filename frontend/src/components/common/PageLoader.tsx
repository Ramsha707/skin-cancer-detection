import { Loader2 } from "lucide-react";

export function PageLoader({ label = "Loading module" }: { label?: string }) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3">
      <Loader2 className="h-6 w-6 animate-spin text-accent-400" />
      <p className="text-xs uppercase tracking-widest text-navy-500">{label}</p>
    </div>
  );
}

/** Inline skeleton block used while a specific panel loads. */
export function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-white/5 ${className}`} />;
}