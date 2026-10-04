import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-accent-500/15 text-accent-300",
        success: "border-transparent bg-emerald-500/15 text-emerald-300",
        warning: "border-transparent bg-amber-500/15 text-amber-300",
        destructive: "border-transparent bg-red-500/15 text-red-300",
        info: "border-transparent bg-sky-500/15 text-sky-300",
        neutral: "border-white/15 bg-white/5 text-navy-200",
        outline: "border-slate-300 bg-white text-slate-700",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

/** Small animated dot used inside status badges. */
function StatusDot({ tone = "emerald", pulse }: { tone?: string; pulse?: boolean }) {
  const colors: Record<string, string> = {
    emerald: "bg-emerald-400",
    amber: "bg-amber-400",
    red: "bg-red-400",
    sky: "bg-sky-400",
    slate: "bg-slate-400",
    accent: "bg-accent-400",
  };
  return (
    <span className="relative flex h-2 w-2">
      {pulse && (
        <span
          className={cn(
            "absolute inline-flex h-full w-full animate-ping rounded-full opacity-60",
            colors[tone] ?? colors.accent,
          )}
        />
      )}
      <span
        className={cn(
          "relative inline-flex h-2 w-2 rounded-full",
          colors[tone] ?? colors.accent,
        )}
      />
    </span>
  );
}

export { Badge, badgeVariants, StatusDot };