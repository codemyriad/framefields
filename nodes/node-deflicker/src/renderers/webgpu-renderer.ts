/// <reference types="webgpu" />

import {
	resolveFrameNumber,
	type WebGPUNodeRenderer,
} from "@framefields/node-sdk";
import {
	drawChildToTexture,
	drawTextureToTarget,
} from "@framefields/node-sdk/gpu";
import { TemporalDeflickerPipeline } from "@framefields/tensor-webgpu";

interface TemporalDeflickerOp {
	op: "TemporalDeflicker" | "Deflicker";
	id?: string;
	blendWeight?: unknown;
	disocclusionThreshold?: unknown;
	maxMotionPixels?: unknown;
	scale?: unknown;
	windowSize?: unknown;
	opacity?: number;
	blendWeightHandleId?: string | null;
	disocclusionThresholdHandleId?: string | null;
	maxMotionPixelsHandleId?: string | null;
	inputs?: Record<
		string,
		{
			connectionValid: boolean;
			outputItem: {
				type: string;
				data: unknown;
			} | null;
		}
	>;
}

function resolveProp(
	op: TemporalDeflickerOp,
	fieldName:
		| "blendWeight"
		| "disocclusionThreshold"
		| "maxMotionPixels"
		| "scale"
		| "windowSize",
	defaultValue: number,
	minVal: number,
	maxVal: number,
	frame: number,
	fps: number,
): number {
	const handleIdKey = `${fieldName}HandleId` as keyof TemporalDeflickerOp;
	const handleId = op[handleIdKey] as string | null | undefined;
	const input = handleId ? op.inputs?.[handleId] : null;

	let raw: unknown;
	if (input?.connectionValid && input.outputItem) {
		const outData = input.outputItem.data as Record<string, unknown> | null;
		raw =
			outData && typeof outData === "object" && "offset" in outData
				? outData.offset
				: input.outputItem.data;
	} else {
		raw = op[fieldName];
	}

	const resolved = resolveFrameNumber(raw, defaultValue, frame, fps);
	return Math.max(minVal, Math.min(maxVal, resolved));
}

const pipelineCache = new Map<string, TemporalDeflickerPipeline>();

function getOrCreatePipeline(
	device: GPUDevice,
	key: string,
): TemporalDeflickerPipeline {
	let pipe = pipelineCache.get(key);
	if (!pipe) {
		pipe = new TemporalDeflickerPipeline(device);
		pipelineCache.set(key, pipe);
	}
	return pipe;
}

export const TemporalDeflickerWebGPURenderer: WebGPUNodeRenderer = async (
	args,
) => {
	const { ctx, encoder, pass, targetWidth, targetHeight, props } = args;

	const op = props.virtualMedia?.operation as TemporalDeflickerOp | undefined;
	if (!op || (op.op !== "TemporalDeflicker" && op.op !== "Deflicker")) {
		return;
	}

	const sourceMedia = props.virtualMedia?.children?.[0];
	if (!sourceMedia) {
		return;
	}

	// End the initial load pass
	pass.end();

	const width = targetWidth;
	const height = targetHeight;
	const frame = props.frame ?? 0;
	const compositionFrame = props.compositionFrame ?? frame;
	const fps = props.fps || 30;

	// 1. Draw child source media into a clean temporary texture
	const { texture: srcTex } = await drawChildToTexture(args, sourceMedia, {
		props: {
			...props,
			virtualMedia: sourceMedia,
			containerWidth: width,
			containerHeight: height,
		},
	});

	// End active pass from child drawing
	args.pass.end();

	// 2. Resolve properties defensively with signal support
	const blendWeight = resolveProp(
		op,
		"blendWeight",
		0.35,
		0.0,
		1.0,
		compositionFrame,
		fps,
	);
	const disocclusionThreshold = resolveProp(
		op,
		"disocclusionThreshold",
		0.15,
		0.01,
		1.0,
		compositionFrame,
		fps,
	);
	const maxMotionPixels = resolveProp(
		op,
		"maxMotionPixels",
		64.0,
		1.0,
		256.0,
		compositionFrame,
		fps,
	);
	const scale = resolveProp(op, "scale", 0.5, 0.1, 1.0, compositionFrame, fps);
	const windowSize = Math.round(
		resolveProp(op, "windowSize", 2, 1, 5, compositionFrame, fps),
	);

	// 3. Execute temporal optical flow warping & adaptive blend
	const pipelineKey =
		props.virtualMedia?.metadata?.id ||
		op.id ||
		`deflicker_${props.renderId ?? "global"}`;
	const deflickerPipe = getOrCreatePipeline(ctx.device, pipelineKey);
	if (frame === 0) {
		deflickerPipe.reset();
	}

	const deflickeredTex = deflickerPipe.execute(
		srcTex,
		{
			blendWeight,
			disocclusionThreshold,
			maxMotionPixels,
			scale,
			windowSize,
		},
		encoder,
	);

	// 4. Render stabilized output to targetView
	const opacity = typeof op.opacity === "number" ? op.opacity : 1.0;
	drawTextureToTarget(args, deflickeredTex, opacity);
};
