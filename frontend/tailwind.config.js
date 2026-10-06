/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: { "2xl": "1400px" },
    },
    extend: {
      colors: {
        /* Purple-tinted neutral ink ramp for light theme text — dark enough
           to hold contrast on white, with the faintest tiers still readable. */
        ink: {
          50: "#f6f5fb",
          100: "#eceaf5",
          200: "#d9d4ea",
          300: "#b5aecb",
          400: "#857ea6",
          500: "#685f8c",
          600: "#524b70",
          700: "#433d5c",
          800: "#353049",
          900: "#2b273f",
          950: "#221f33",
        },
        /* Vibrant pastel families: blush pink, lilac purple, periwinkle blue */
        blush: {
          50: "#fff0f6",
          100: "#ffe3ee",
          200: "#ffc7de",
          300: "#ffa2c9",
          400: "#fb6cab",
          500: "#f23d92",
        },
        lilac: {
          50: "#f6f2ff",
          100: "#ede5ff",
          200: "#ddccff",
          300: "#c3a7fd",
          400: "#a377f9",
          500: "#8b4ff0",
        },
        periwinkle: {
          50: "#f1f5ff",
          100: "#e4edff",
          200: "#ccdcff",
          300: "#aac2ff",
          400: "#8aa0fb",
          500: "#6781f2",
        },
        /* Primary accent — vivid violet (aligned to Tailwind violet scale) */
        accent: {
          50: "#f5f3ff",
          100: "#ede9fe",
          200: "#ddd6fe",
          300: "#c4b5fd",
          400: "#a78bfa",
          500: "#8b5cf6",
          600: "#7c3aed",
          700: "#6d28d9",
          800: "#5b21b6",
          900: "#4c1d95",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent_prim: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        soft: "0 1px 2px 0 rgb(43 39 63 / 0.05), 0 6px 24px -6px rgb(96 165 250 / 0.22)",
        glow: "0 0 0 1px rgb(191 219 254 / 0.5), 0 8px 32px -8px rgb(96 165 250 / 0.45)",
        "glow-btn": "0 2px 10px -2px rgb(96 165 250 / 0.5), 0 8px 24px -6px rgb(251 113 133 / 0.4)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        pulseRing: {
          "0%": { boxShadow: "0 0 0 0 rgb(96 165 250 / 0.5)" },
          "70%": { boxShadow: "0 0 0 12px rgb(96 165 250 / 0)" },
          "100%": { boxShadow: "0 0 0 0 rgb(96 165 250 / 0)" },
        },
        dashFlow: {
          to: { strokeDashoffset: "-1000" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-8px)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "pulse-ring": "pulseRing 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "dash-flow": "dashFlow 3s linear infinite",
        float: "float 6s ease-in-out infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};