import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Proxy /api to the FastAPI backend (configs/default.yaml api.port 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
