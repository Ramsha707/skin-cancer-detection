import * as React from "react";
import { cn } from "@/lib/utils";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number | null;
  /** 0-100 */
  max?: number;
  indicatorClassName?: string;
}

export function Progress({
  value,
  max = 100,
  className,
  indicatorClassName,
  ...props
}: ProgressProps) {
  const pct = value === null || value === undefined ? 0 : Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div
      role="progressbar"
      aria-valuenow={value ?? undefined}
      aria-valuemin={0}
      aria-valuemax={max}
      className={cn("relative h-2 w-full overflow-hidden rounded-full bg-slate-200", className)}
      {...props}
    >
      <div
        className={cn(
          "h-full rounded-full bg-gradient-to-r from-accent-500 to-accent-400 transition-all duration-500",
          indicatorClassName,
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

/** Dark-surface variant for panels sitting on the navy shell. */
export function ProgressDark({
  value,
  max = 100,
  className,
  indicatorClassName,
  color,
}: ProgressProps & {
  /** Solid fill colour, e.g. a per-class hex from the taxonomy. */
  color?: string;
}) {
  const pct = value === null || value === undefined ? 0 : Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-white/10", className)}>
      <div
        className={cn(
          "h-full rounded-full bg-gradient-to-r from-accent-400 to-accent-300 transition-all duration-500",
          !color && "bg-gradient-to-r from-accent-400 to-accent-300",
          color && "bg-none",
          indicatorClassName,
        )}
        style={{ width: `${pct}%`, ...(color ? { backgroundColor: color } : {}) }}
      />
    </div>
  );
}