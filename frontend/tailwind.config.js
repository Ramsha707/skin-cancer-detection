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
        /* Purple-tinted neutral ink ramp for light theme text */
        ink: {
          50: "#f7f5fb",
          100: "#efecf6",
          200: "#ddd9e9",
          300: "#c3bdd6",
          400: "#8f87a8",
          500: "#7d7496",
          600: "#625a7e",
          700: "#514a6e",
          800: "#413b59",
          900: "#35304a",
          950: "#292339",
        },
        /* Soft pastel surfaces: blush pink, lilac purple, periwinkle blue */
        blush: {
          50: "#fef2f7",
          100: "#fde6ef",
          200: "#fbcfe2",
          300: "#f9a8cd",
          400: "#f47eb0",
          500: "#ec5a93",
        },
        lilac: {
          50: "#f6f3ff",
          100: "#ede7fe",
          200: "#ddd4fc",
          300: "#c4b5fb",
          400: "#a78bfa",
          500: "#8b6ef4",
        },
        periwinkle: {
          50: "#f0f5ff",
          100: "#e2ebfe",
          200: "#c9d9fd",
          300: "#a5c0fb",
          400: "#7da3f7",
          500: "#5b86ef",
        },
        /* Primary accent — soft violet that reads on light surfaces */
        accent: {
          50: "#f4f1fe",
          100: "#ebe5fd",
          200: "#d9d0fb",
          300: "#8b6ff0",
          400: "#7857e8",
          500: "#6741db",
          600: "#5833c2",
          700: "#4a2aa5",
          800: "#3c2287",
          900: "#311d6c",
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
        soft: "0 1px 2px 0 rgb(41 35 57 / 0.04), 0 4px 16px -2px rgb(103 65 219 / 0.08)",
        glow: "0 0 0 1px rgb(167 139 250 / 0.35), 0 8px 32px -8px rgb(139 110 244 / 0.35)",
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
          "0%": { boxShadow: "0 0 0 0 rgb(167 139 250 / 0.45)" },
          "70%": { boxShadow: "0 0 0 12px rgb(167 139 250 / 0)" },
          "100%": { boxShadow: "0 0 0 0 rgb(167 139 250 / 0)" },
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