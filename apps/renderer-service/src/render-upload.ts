import type { VirtualMediaData } from "@framefields/core";
import { HeadlessMediaRenderer } from "@framefields/renderer";
import {
	container,
	rendererLogger,
	type StorageService,
	TOKENS,
} from "@framefields/server-utils";

export interface RenderUploadOptions {
	virtualMedia: VirtualMediaData;
	/** "video" / "gif" render a file, "lut" a LUT, anything else a PNG frame. */
	type: string;
	frame?: number;
	codec?: string;
	audioCodec?: string;
	quality?: string;
	key: string;
	bucket: string;
	/** Log line prefix, e.g. "[CLI]". */
	logPrefix: string;
}

function videoContentType(codec: string | undefined): string {
	return codec === "vp8" || codec === "vp9"
		? "video/webm"
		: codec === "gif"
			? "image/gif"
			: codec === "mp3"
				? "audio/mpeg"
				: codec === "aac"
					? "audio/mp4"
					: codec === "opus"
						? "audio/webm"
						: "video/mp4";
}

/**
 * Renders `virtualMedia` and uploads the result to storage under `key`.
 * Returns the uploaded content type.
 */
export async function renderAndUpload(
	options: RenderUploadOptions,
): Promise<string> {
	const { virtualMedia, frame, key, bucket, logPrefix } = options;
	const normalizedType = options.type.toLowerCase();
	const renderer = new HeadlessMediaRenderer();
	let buffer: Buffer | undefined;
	let filePath: string | undefined;
	let contentType: string;
	let fileCleanup: (() => Promise<void>) | undefined;

	if (normalizedType === "video" || normalizedType === "gif") {
		const renderResult = await renderer.renderVideo(virtualMedia, {
			codec: options.codec,
			audioCodec: options.audioCodec,
			quality: options.quality,
		});
		filePath = renderResult.filePath;
		fileCleanup = renderResult.cleanup;
		contentType = videoContentType(options.codec);
	} else if (normalizedType === "lut") {
		buffer = await renderer.renderLut(virtualMedia, frame);
		contentType = "text/plain";
	} else {
		buffer = await renderer.renderImage(virtualMedia, frame);
		contentType = "image/png";
	}

	const storage = container.get<StorageService>(TOKENS.STORAGE);
	if (filePath) {
		rendererLogger.info(
			{ filePath },
			`${logPrefix} Render complete - uploading file to storage.`,
		);
		await storage.uploadFileToStorage(filePath, key, contentType, bucket);
	} else if (buffer) {
		rendererLogger.info(
			{ fileSize: buffer.length },
			`${logPrefix} Render complete - uploading buffer to storage.`,
		);
		await storage.uploadToStorage(buffer, key, contentType, bucket);
	}

	if (fileCleanup) {
		await fileCleanup().catch(() => {});
	}
	return contentType;
}
