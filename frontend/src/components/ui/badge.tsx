import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-transparent bg-violet-200 text-violet-900",
        success: "border-transparent bg-emerald-200 text-emerald-800",
        warning: "border-transparent bg-amber-200 text-amber-800",
        destructive: "border-transparent bg-red-200 text-red-800",
        info: "border-transparent bg-sky-200 text-sky-800",
        neutral: "border-ink-950/15 bg-ink-950/5 text-ink-700",
        outline: "border-lilac-300 bg-white text-ink-700",
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
    emerald: "bg-emerald-500",
    amber: "bg-amber-500",
    red: "bg-red-500",
    sky: "bg-sky-500",
    slate: "bg-ink-500",
    accent: "bg-accent-500",
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