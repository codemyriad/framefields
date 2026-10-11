import type { VirtualMediaData } from "@framefields/core";
import { rendererLogger } from "@framefields/server-utils";
import { renderAndUpload } from "./render-upload.js";

export interface RenderJobInput {
	virtualMedia?: VirtualMediaData;
	type: "Video" | "Image" | "LUT";
	frame?: number;
	fps?: number;
	width?: number;
	height?: number;
	bucket: string;
	key: string;
	codec?: string;
	audioCodec?: string;
	quality?: string;
}

export interface RenderJob {
	id: string;
	status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED";
	output?: { success: boolean; key: string; error?: string };
	error?: string;
}

export const jobs = new Map<string, RenderJob>();

export const hasActiveJobs = (): boolean => {
	for (const job of jobs.values()) {
		if (job.status === "IN_PROGRESS" || job.status === "IN_QUEUE") {
			return true;
		}
	}
	return false;
};

export async function executeRenderJob(
	jobId: string,
	jobData: RenderJobInput,
	onStateChange?: (state: "processing" | "idle") => void,
): Promise<void> {
	const job = jobs.get(jobId);
	if (!job) return;

	try {
		onStateChange?.("processing");
		job.status = "IN_PROGRESS";
		const {
			virtualMedia,
			type,
			frame,
			bucket,
			key,
			codec,
			audioCodec,
			quality,
		} = jobData;

		let probedMeta: any;

		const normalizedType = type.toLowerCase();
		rendererLogger.info(
			{ type: normalizedType, key },
			"[HTTP Local] Starting render process...",
		);

		// Standard Virtual Media Rendering path - enforce metadata presence
		if (!virtualMedia) {
			throw new Error("virtualMedia is required for standard media rendering.");
		}
		if (!virtualMedia.metadata) {
			throw new Error(
				"virtualMedia.metadata is required for standard media rendering.",
			);
		}

		await renderAndUpload({
			virtualMedia,
			type,
			frame,
			codec,
			audioCodec,
			quality,
			key,
			bucket,
			logPrefix: "[HTTP Local]",
		});

		job.status = "COMPLETED";
		job.output = {
			success: true,
			key,
			metadata: probedMeta,
		} as any;
		rendererLogger.info(
			{ jobId, key },
			"[HTTP Local] Render complete and uploaded successfully.",
		);
	} catch (err) {
		const errMsg = err instanceof Error ? err.message : String(err);
		rendererLogger.error(
			{ jobId, err: errMsg },
			"[HTTP Local] Rendering job failed.",
		);
		job.status = "FAILED";
		job.error = errMsg;
	} finally {
		if (!hasActiveJobs()) {
			onStateChange?.("idle");
		}
	}
}
