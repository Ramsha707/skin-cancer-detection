import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Keeps the browser same-origin in dev so no CORS config is needed.
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // This Rollup version only accepts the function form of manualChunks.
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("recharts") || id.includes("d3-")) return "charts";
          if (id.includes("framer-motion") || id.includes("motion-dom") || id.includes("motion-utils"))
            return "motion";
          if (
            id.includes("react-router") ||
            id.includes("/react-dom/") ||
            id.includes("/react/") ||
            id.includes("scheduler")
          )
            return "vendor";
          return undefined;
        },
      },
    },
  },
});