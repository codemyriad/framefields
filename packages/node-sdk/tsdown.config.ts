import { defineConfig } from "tsdown";

export default defineConfig({
	entry: {
		index: "index.ts",
		renderer: "renderer/index.ts",
		gpu: "renderer/gpu.ts",
		testing: "testing/index.ts",
	},
	format: ["esm"],
	dts: true,
	clean: true,
});
