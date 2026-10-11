import "reflect-metadata";
import { configureAssetUrls } from "@framefields/client-utils";
import { HeadlessMediaRenderer } from "@framefields/renderer";
import "@framefields/renderer/offscreen-gl-polyfill";
import {
	container,
	rendererLogger,
	type StorageService,
	TOKENS,
} from "@framefields/server-utils";
import { registerBackendServices } from "./di-setup.js";
import { RENDERER_ENV_CONFIG } from "./env-config.js";
import { renderAndUpload } from "./render-upload.js";

async function main() {
	const jobArgIndex = process.argv.indexOf("--job");
	if (jobArgIndex === -1 || !process.argv[jobArgIndex + 1]) {
		console.error("Missing --job parameter containing JSON input payload");
		process.exit(1);
	}

	let jobData: any;
	try {
		jobData = JSON.parse(process.argv[jobArgIndex + 1]);
	} catch (e: any) {
		console.error("Failed to parse JSON job input:", e.message);
		process.exit(1);
	}

	const { virtualMedia, type, frame, bucket, key, codec, audioCodec, quality } =
		jobData;

	if (!virtualMedia || !type || !bucket || !key) {
		console.error(
			"Missing required job data fields (virtualMedia, type, bucket, key)",
		);
		process.exit(1);
	}

	try {
		// Initialize configurations
		configureAssetUrls({
			baseUrl: RENDERER_ENV_CONFIG.BASE_URL,
			r2CustomDomain: RENDERER_ENV_CONFIG.R2_CUSTOM_DOMAIN,
		});

		// Boot DI services
		await registerBackendServices();

		// Eagerly initialize renderer
		await HeadlessMediaRenderer.initialize();

		const normalizedType = type.toLowerCase();
		rendererLogger.info(
			{ type: normalizedType, key },
			"[CLI] Starting render process...",
		);

		let probedMeta: any;

		const contentType = await renderAndUpload({
			virtualMedia,
			type,
			frame,
			codec,
			audioCodec,
			quality,
			key,
			bucket,
			logPrefix: "[CLI]",
		});

		const storage = container.get<StorageService>(TOKENS.STORAGE);
		const publicUrl = storage.getPublicUrl?.(key, bucket);
		rendererLogger.info(
			{ key, bucket, contentType },
			"[CLI] Render complete and uploaded to R2 successfully.",
		);

		// Print the final result in JSON format as stdout for the wrapper script to read
		console.log(
			JSON.stringify({
				success: true,
				key,
				bucket,
				publicUrl,
				metadata: probedMeta,
			}),
		);
		process.exit(0);
	} catch (error: any) {
		rendererLogger.error(
			{ err: error.message || String(error) },
			"[CLI] Rendering job failed.",
		);
		console.error(error.message || String(error));
		process.exit(1);
	}
}

main();
