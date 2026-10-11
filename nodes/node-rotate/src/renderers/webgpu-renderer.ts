/// <reference types="webgpu" />
import type { VirtualMediaData } from "@framefields/core";
import type { WebGPUNodeRenderer } from "@framefields/node-sdk";
import {
	createFullscreenPipeline,
	drawChildToTexture,
	drawFullscreenPass,
	drawTextureToTarget,
	FULLSCREEN_QUAD_WGSL,
	fragmentBindGroupLayout,
} from "@framefields/node-sdk/gpu";
import { probeVideoSize } from "@framefields/webgpu-renderers";
import { rotatedSourceSize } from "../shared/config.js";

/** A plain number, a per-frame function, or a signal (`{ value }`). */
type NumericProp =
	| number
	| ((frame: number, fps: number) => number)
	| { value: unknown };

interface RotateOp {
	op: "Rotate";
	angle?: NumericProp;
	fit?: "cover" | "contain" | "fill";
	scale?: NumericProp;
	opacity?: number;
}

// Inverse-maps each output pixel into the source: p (px from the output
// center) turned back by the angle, divided by the source's on-screen size.
// Edges are antialiased over one output pixel so animated angles stay clean.
const WGSL_ROTATE_SHADER = `
struct RotateUniforms {
	outSize: vec2<f32>,
	size   : vec2<f32>,
	cs     : vec2<f32>,
	_pad   : vec2<f32>,
};

@group(0) @binding(0) var<uniform> u : RotateUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;
${FULLSCREEN_QUAD_WGSL}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let p = (in.uv - vec2<f32>(0.5)) * u.outSize;
	// Clockwise on screen (y down): undo it with the transpose.
	let q = vec2<f32>(u.cs.x * p.x + u.cs.y * p.y, -u.cs.y * p.x + u.cs.x * p.y);
	let uv = q / u.size + vec2<f32>(0.5);
	// Distance (output px) inside the source rectangle's nearest edge.
	let edge = (vec2<f32>(0.5) - abs(uv - vec2<f32>(0.5))) * u.size;
	let coverage = clamp(min(edge.x, edge.y) + 0.5, 0.0, 1.0);
	let color = textureSampleLevel(tex, samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
	// Premultiplied: coverage scales every channel.
	return color * coverage;
}
`;

interface DeviceRotateResources {
	uniformLayout: GPUBindGroupLayout;
	textureLayout: GPUBindGroupLayout;
	pipeline: GPURenderPipeline;
}

const deviceResourceCache = new WeakMap<GPUDevice, DeviceRotateResources>();
const uniformData = new Float32Array(8);

function getRotateResources(
	device: GPUDevice,
	targetFormat: GPUTextureFormat,
): DeviceRotateResources {
	const cached = deviceResourceCache.get(device);
	if (cached) return cached;

	const uniformLayout = fragmentBindGroupLayout(device, ["uniform"]);

	const textureLayout = fragmentBindGroupLayout(device, ["texture", "sampler"]);

	const shaderModule = device.createShaderModule({ code: WGSL_ROTATE_SHADER });

	const pipeline = createFullscreenPipeline(device, {
		module: shaderModule,
		layouts: [uniformLayout, textureLayout],
		format: targetFormat,
	});

	const res: DeviceRotateResources = { uniformLayout, textureLayout, pipeline };
	deviceResourceCache.set(device, res);
	return res;
}

function resolveNumber(
	raw: NumericProp | undefined,
	fallback: number,
	frame: number,
	fps: number,
): number {
	const value =
		typeof raw === "function"
			? raw(frame, fps)
			: raw !== null && typeof raw === "object"
				? "get" in raw && typeof (raw as { get: unknown }).get === "function"
					? (
							raw as { get: (ctx: { frame: number; fps: number }) => unknown }
						).get({ frame, fps })
					: "value" in raw
						? raw.value
						: raw
				: raw;
	const n = Number(value ?? fallback);
	return Number.isFinite(n) ? n : fallback;
}

/**
 * The child's own size: the nearest stated metadata down the chain, else the
 * source file's display size. A layer built from a bare path states none, and
 * guessing the layer box would turn a 16:9 clip into a squeezed 9:16 one.
 */
async function intrinsicSize(
	media: VirtualMediaData,
): Promise<{ width: number; height: number } | null> {
	for (let node: VirtualMediaData | undefined = media; node; ) {
		const { width, height } = node.metadata ?? {};
		if (width && height) return { width, height };
		const op = node.operation as {
			op?: string;
			url?: unknown;
			source?: unknown;
		};
		if (op.op === "source") {
			const url = typeof op.url === "string" ? op.url : op.source;
			return typeof url === "string" ? probeVideoSize(url) : null;
		}
		node = node.children?.[0];
	}
	return null;
}

export const WebGPURenderer: WebGPUNodeRenderer = async (args) => {
	const {
		ctx,
		encoder,
		pass,
		targetTexture,
		targetWidth,
		targetHeight,
		props,
	} = args;

	const op = props.virtualMedia?.operation as RotateOp | undefined;
	if (op?.op !== "Rotate" || !op) return;

	const childMedia = props.virtualMedia.children?.[0];
	if (!childMedia) return;

	pass.end();

	const frame = props.frame ?? 0;
	const compositionFrame = props.compositionFrame ?? frame;
	const fps = props.fps || 30;
	const width = targetWidth;
	const height = targetHeight;

	const intrinsic = await intrinsicSize(childMedia);
	const sourceWidth = intrinsic?.width ?? props.containerWidth ?? width;
	const sourceHeight = intrinsic?.height ?? props.containerHeight ?? height;

	// 1. Render the child upright into an offscreen texture at its own size
	const { texture: childTex } = await drawChildToTexture(args, childMedia, {
		width: sourceWidth,
		height: sourceHeight,
		props: {
			...props,
			containerWidth: sourceWidth,
			containerHeight: sourceHeight,
			virtualMedia: childMedia,
		},
	});

	// 2. Rotate it into the output
	const angle = resolveNumber(op.angle, 90, compositionFrame, fps);
	const scale = Math.max(
		0.01,
		resolveNumber(op.scale, 1, compositionFrame, fps),
	);
	const size = rotatedSourceSize(
		sourceWidth,
		sourceHeight,
		width,
		height,
		angle,
		op.fit ?? "cover",
	);
	const rad = (angle * Math.PI) / 180;

	uniformData[0] = width;
	uniformData[1] = height;
	uniformData[2] = size.width * scale;
	uniformData[3] = size.height * scale;
	uniformData[4] = Math.cos(rad);
	uniformData[5] = Math.sin(rad);

	const { pipeline, uniformLayout, textureLayout } = getRotateResources(
		ctx.device,
		ctx.renderer.format,
	);

	const uBindGroup = ctx.device.createBindGroup({
		layout: uniformLayout,
		entries: [
			{
				binding: 0,
				resource: { buffer: ctx.renderer.getTemporaryBuffer(uniformData) },
			},
		],
	});

	const tBindGroup = ctx.device.createBindGroup({
		layout: textureLayout,
		entries: [
			{ binding: 0, resource: childTex.createView() },
			{
				binding: 1,
				resource: ctx.renderer.samplerCache.getSampler(ctx.device),
			},
		],
	});

	const outTex = ctx.renderer.getTemporaryTexture(width, height, [
		childTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);

	drawFullscreenPass(encoder, outTex.createView(), pipeline, [
		uBindGroup,
		tBindGroup,
	]);

	drawTextureToTarget(args, outTex, op.opacity ?? 1.0);
};
