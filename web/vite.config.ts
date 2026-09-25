import { defineConfig } from "vite";
import { cpSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";

const siteAssets = fileURLToPath(new URL("../site_assets", import.meta.url));
const deployedSiteAssets = fileURLToPath(new URL("./dist/site_assets", import.meta.url));

export default defineConfig({
  base: "/wiuthackathon/",
  plugins: [{
    name: "copy-site-assets",
    closeBundle() {
      rmSync(deployedSiteAssets, { recursive: true, force: true });
      cpSync(siteAssets, deployedSiteAssets, { recursive: true });
    },
  }],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
});
