/// <reference types="webgpu" />
import type { SignalData } from "@framefields/core";
import type { WebGPUNodeRenderer } from "@framefields/node-sdk";
import {
	createFullscreenPipeline,
	drawChildToTexture,
	drawFullscreenPass,
	drawTextureToTarget,
	FULLSCREEN_QUAD_WGSL,
	fragmentBindGroupLayout,
	signalTiming,
} from "@framefields/node-sdk/gpu";
import { parseColor, signalRegistry } from "@framefields/webgpu-renderers";
import type { GradientMapOperation, GradientStop } from "../shared/config.js";

type GradientMapSignalData = SignalData & { nodeId?: string };

interface GradientMapOp extends GradientMapOperation {
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

interface DeviceGradientMapResources {
	pipeline: GPURenderPipeline;
	uniformLayout: GPUBindGroupLayout;
	textureLayout: GPUBindGroupLayout;
	signalLayout: GPUBindGroupLayout;
}

const deviceResourceCache = new WeakMap<
	GPUDevice,
	DeviceGradientMapResources
>();

const WGSL_GRADIENT_MAP_SHADER = `
struct GradientStop {
	pos : vec4<f32>,
	col : vec4<f32>,
};

struct GradientMapUniforms {
	numStops      : f32,
	isSmooth      : f32,
	dither        : f32,
	opacity       : f32,

	hasOpacitySig : f32,
	pad0          : f32,
	pad1          : f32,
	pad2          : f32,

	stops         : array<GradientStop, 16>,
};

@group(0) @binding(0) var<uniform> u : GradientMapUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;
@group(2) @binding(0) var opacitySigTex : texture_2d<f32>;
@group(2) @binding(1) var signalSamp    : sampler;
${FULLSCREEN_QUAD_WGSL}

fn ditherNoise(pos: vec2<f32>) -> f32 {
	let p = fract(pos * vec2<f32>(0.1031, 0.1030));
	let d = dot(p, p.yx + 33.33);
	return fract((p.x + p.y) * d) - 0.5;
}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let color = textureSampleLevel(tex, samp, in.uv, 0.0);
	if (color.a <= 0.00001) {
		return color;
	}

	let alpha = color.a;
	let rgb = color.rgb / max(alpha, 0.0001);

	// ITU-R BT.709 Luminance
	var lum = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

	if (u.dither > 0.5) {
		let d = ditherNoise(in.pos.xy) * (1.0 / 255.0);
		lum = clamp(lum + d, 0.0, 1.0);
	} else {
		lum = clamp(lum, 0.0, 1.0);
	}

	let count = i32(clamp(u.numStops, 2.0, 16.0));
	var gradColor = u.stops[0].col.rgb;

	if (lum <= u.stops[0].pos.x) {
		gradColor = u.stops[0].col.rgb;
	} else if (lum >= u.stops[count - 1].pos.x) {
		gradColor = u.stops[count - 1].col.rgb;
	} else {
		for (var i = 0; i < 15; i = i + 1) {
			if (i >= count - 1) {
				break;
			}
			let p0 = u.stops[i].pos.x;
			let p1 = u.stops[i + 1].pos.x;

			if (lum >= p0 && lum <= p1) {
				let range = max(p1 - p0, 1e-5);
				let t = clamp((lum - p0) / range, 0.0, 1.0);
				if (u.isSmooth > 0.5) {
					gradColor = mix(u.stops[i].col.rgb, u.stops[i + 1].col.rgb, t);
				} else {
					gradColor = u.stops[i].col.rgb;
				}
				break;
			}
		}
	}

	var opacity = u.opacity;
	if (u.hasOpacitySig > 0.5) {
		opacity = textureSampleLevel(opacitySigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	opacity = clamp(opacity, 0.0, 1.0);

	let finalRgb = mix(rgb, gradColor, opacity);
	return vec4<f32>(finalRgb * alpha, alpha);
}
`;

function getGradientMapResources(
	device: GPUDevice,
	format: GPUTextureFormat,
): DeviceGradientMapResources {
	let res = deviceResourceCache.get(device);
	if (res) return res;

	const uniformLayout = fragmentBindGroupLayout(
		device,
		["uniform"],
		"GradientMapUniformLayout",
	);

	const textureLayout = fragmentBindGroupLayout(
		device,
		["texture", "sampler"],
		"GradientMapTextureLayout",
	);

	const signalLayout = fragmentBindGroupLayout(
		device,
		["texture", "sampler"],
		"GradientMapSignalLayout",
	);

	const module = device.createShaderModule({
		label: "GradientMapShaderModule",
		code: WGSL_GRADIENT_MAP_SHADER,
	});

	const pipeline = createFullscreenPipeline(device, {
		label: "GradientMapPipeline",
		module,
		layouts: [uniformLayout, textureLayout, signalLayout],
		format,
	});

	res = {
		pipeline,
		uniformLayout,
		textureLayout,
		signalLayout,
	};
	deviceResourceCache.set(device, res);
	return res;
}

const uniformBufferData = new Float32Array(136);

export const GradientMapWebGPURenderer: WebGPUNodeRenderer = async (args) => {
	const {
		ctx,
		encoder,
		pass,
		targetTexture,
		targetWidth,
		targetHeight,
		props,
	} = args;

	const frame = props.frame ?? 0;
	const fps = props.fps || 30;

	const op = props.virtualMedia?.operation as unknown as
		| GradientMapOp
		| undefined;

	if (op?.op !== "GradientMap" || !props.virtualMedia?.children?.[0]) return;

	pass.end();

	const childMedia = props.virtualMedia.children[0];
	const width = targetWidth;
	const height = targetHeight;

	const { texture: tmpTex } = await drawChildToTexture(args, childMedia);

	// Parse stops
	const rawStops: GradientStop[] = Array.isArray(op.stops)
		? op.stops
		: [
				{ position: 0.0, color: "#000000" },
				{ position: 1.0, color: "#ffffff" },
			];

	const sortedStops = [...rawStops].sort((a, b) => a.position - b.position);
	const numStops = Math.min(Math.max(sortedStops.length, 2), 16);

	// Resolve opacity signal
	const opacityHandleId = op.opacityHandleId;
	const opacitySignalInput = opacityHandleId
		? op.inputs?.[opacityHandleId]
		: null;

	const hasOpacitySig = !!(
		opacitySignalInput?.connectionValid &&
		opacitySignalInput.outputItem?.type === "Signal"
	);
	const opacitySigData = (
		hasOpacitySig ? opacitySignalInput?.outputItem?.data : null
	) as GradientMapSignalData | null;

	let opacityVal = Number(op.opacity ?? 1.0);
	if (
		!hasOpacitySig &&
		opacitySignalInput?.connectionValid &&
		opacitySignalInput.outputItem?.type === "Number"
	) {
		opacityVal = Number(opacitySignalInput.outputItem.data ?? 1.0);
	}

	const { elapsedSeconds, durationSeconds } = signalTiming(props, frame, fps);

	let opacityView: GPUTextureView;
	if (hasOpacitySig && opacitySigData) {
		opacityView = signalRegistry.getOrCreate2DTextureView(
			ctx.device,
			encoder,
			opacitySigData.nodeId ?? "gradient_map_opacity",
			elapsedSeconds,
			durationSeconds,
			opacitySigData,
			width,
			height,
			props.renderId,
			frame,
			fps,
		);
	} else {
		opacityView = signalRegistry.getDummy1x1TextureView(ctx.device);
	}

	// Build uniform data (136 floats = 544 bytes)
	uniformBufferData.fill(0);
	uniformBufferData[0] = numStops;
	uniformBufferData[1] = op.smooth === false ? 0.0 : 1.0;
	uniformBufferData[2] = op.dither === false ? 0.0 : 1.0;
	uniformBufferData[3] = opacityVal;
	uniformBufferData[4] = hasOpacitySig ? 1.0 : 0.0;
	uniformBufferData[5] = 0.0;
	uniformBufferData[6] = 0.0;
	uniformBufferData[7] = 0.0;

	for (let i = 0; i < 16; i++) {
		const stop = sortedStops[Math.min(i, numStops - 1)];
		const baseIdx = 8 + i * 8;
		const parsed = parseColor(stop.color);
		// pos: vec4<f32>
		uniformBufferData[baseIdx + 0] = stop.position;
		uniformBufferData[baseIdx + 1] = 0.0;
		uniformBufferData[baseIdx + 2] = 0.0;
		uniformBufferData[baseIdx + 3] = 0.0;
		// col: vec4<f32>
		uniformBufferData[baseIdx + 4] = parsed.r;
		uniformBufferData[baseIdx + 5] = parsed.g;
		uniformBufferData[baseIdx + 6] = parsed.b;
		uniformBufferData[baseIdx + 7] = parsed.a;
	}

	const { pipeline, uniformLayout, textureLayout, signalLayout } =
		getGradientMapResources(ctx.device, ctx.renderer.format);

	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);
	const buffer = ctx.renderer.getTemporaryBuffer(uniformBufferData);

	const outTex = ctx.renderer.getTemporaryTexture(width, height, [
		tmpTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);

	drawFullscreenPass(encoder, outTex.createView(), pipeline, [
		ctx.device.createBindGroup({
			layout: uniformLayout,
			entries: [{ binding: 0, resource: { buffer } }],
		}),
		ctx.renderer.bindGroupCache.getBindGroup(
			ctx.device,
			textureLayout,
			tmpTex,
			sampler,
		),
		ctx.device.createBindGroup({
			layout: signalLayout,
			entries: [
				{ binding: 0, resource: opacityView },
				{ binding: 1, resource: sampler },
			],
		}),
	]);

	drawTextureToTarget(args, outTex, props.opacity ?? 1);
};
