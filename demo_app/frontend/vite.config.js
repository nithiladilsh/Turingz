import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // allow importing committed result JSON from the repo's results/ folder (one source of truth)
  server: { port: 5173, fs: { allow: ["../../"] } },
});
