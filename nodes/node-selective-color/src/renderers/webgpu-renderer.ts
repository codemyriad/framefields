/// <reference types="webgpu" />
import type { WebGPUNodeRenderer } from "@framefields/node-sdk";
import {
	createFullscreenPipeline,
	drawChildToTexture,
	drawFullscreenPass,
	drawTextureToTarget,
	FULLSCREEN_QUAD_WGSL,
	fragmentBindGroupLayout,
} from "@framefields/node-sdk/gpu";
import type {
	ColorAdjustment,
	SelectiveColorOperation,
} from "../shared/config.js";

interface DeviceSelectiveColorResources {
	pipeline: GPURenderPipeline;
	uniformLayout: GPUBindGroupLayout;
	textureLayout: GPUBindGroupLayout;
}

const deviceResourceCache = new WeakMap<
	GPUDevice,
	DeviceSelectiveColorResources
>();

const WGSL_SELECTIVE_COLOR_SHADER = `
struct SelectiveColorUniforms {
	reds     : vec4<f32>,
	yellows  : vec4<f32>,
	greens   : vec4<f32>,
	cyans    : vec4<f32>,
	blues    : vec4<f32>,
	magentas : vec4<f32>,
	whites   : vec4<f32>,
	neutrals : vec4<f32>,
	blacks   : vec4<f32>,
	method   : f32,
	pad0     : f32,
	pad1     : f32,
	pad2     : f32,
};

@group(0) @binding(0) var<uniform> u : SelectiveColorUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;
${FULLSCREEN_QUAD_WGSL}

fn calcAdjustment(cmy: vec3<f32>, adj: vec4<f32>, weight: f32, isAbsolute: bool) -> vec3<f32> {
	if (weight <= 0.00001) {
		return vec3<f32>(0.0);
	}
	let adjC = adj.x;
	let adjM = adj.y;
	let adjY = adj.z;
	let adjK = adj.w;

	if (isAbsolute) {
		let cDelta = adjC + adjK * (1.0 + adjC);
		let mDelta = adjM + adjK * (1.0 + adjM);
		let yDelta = adjY + adjK * (1.0 + adjY);
		return vec3<f32>(cDelta, mDelta, yDelta) * weight;
	} else {
		let cDelta = ((1.0 + adjK) * (1.0 + adjC) - 1.0) * cmy.x;
		let mDelta = ((1.0 + adjK) * (1.0 + adjM) - 1.0) * cmy.y;
		let yDelta = ((1.0 + adjK) * (1.0 + adjY) - 1.0) * cmy.z;
		return vec3<f32>(cDelta, mDelta, yDelta) * weight;
	}
}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let color = textureSampleLevel(tex, samp, in.uv, 0.0);
	if (color.a <= 0.00001) {
		return color;
	}

	let rgb = clamp(color.rgb / max(color.a, 0.00001), vec3<f32>(0.0), vec3<f32>(1.0));
	let r = rgb.r;
	let g = rgb.g;
	let b = rgb.b;

	let maxVal = max(r, max(g, b));
	let minVal = min(r, min(g, b));

	// Chromatic weights
	let wRed     = max(0.0, r - max(g, b));
	let wYellow  = max(0.0, min(r, g) - b);
	let wGreen   = max(0.0, g - max(r, b));
	let wCyan    = max(0.0, min(g, b) - r);
	let wBlue    = max(0.0, b - max(r, g));
	let wMagenta = max(0.0, min(r, b) - g);

	// Achromatic weights (Whites, Blacks, and true Midtone Neutrals)
	let wWhite   = max(0.0, (minVal - 0.5) * 2.0);
	let wBlack   = max(0.0, (0.5 - maxVal) * 2.0);
	let wNeutral = max(0.0, 1.0 - (wRed + wYellow + wGreen + wCyan + wBlue + wMagenta + wWhite + wBlack));

	let cmy = vec3<f32>(1.0 - r, 1.0 - g, 1.0 - b);
	let isAbs = u.method > 0.5;

	var totalDelta = vec3<f32>(0.0);
	totalDelta += calcAdjustment(cmy, u.reds, wRed, isAbs);
	totalDelta += calcAdjustment(cmy, u.yellows, wYellow, isAbs);
	totalDelta += calcAdjustment(cmy, u.greens, wGreen, isAbs);
	totalDelta += calcAdjustment(cmy, u.cyans, wCyan, isAbs);
	totalDelta += calcAdjustment(cmy, u.blues, wBlue, isAbs);
	totalDelta += calcAdjustment(cmy, u.magentas, wMagenta, isAbs);
	totalDelta += calcAdjustment(cmy, u.whites, wWhite, isAbs);
	totalDelta += calcAdjustment(cmy, u.neutrals, wNeutral, isAbs);
	totalDelta += calcAdjustment(cmy, u.blacks, wBlack, isAbs);

	let finalCMY = clamp(cmy + totalDelta, vec3<f32>(0.0), vec3<f32>(1.0));
	let finalRGB = vec3<f32>(1.0 - finalCMY.x, 1.0 - finalCMY.y, 1.0 - finalCMY.z);

	return vec4<f32>(finalRGB * color.a, color.a);
}
`;

function getSelectiveColorResources(
	device: GPUDevice,
	format: GPUTextureFormat,
): DeviceSelectiveColorResources {
	let res = deviceResourceCache.get(device);
	if (!res) {
		const uniformLayout = fragmentBindGroupLayout(device, ["uniform"]);

		const textureLayout = fragmentBindGroupLayout(device, [
			"texture",
			"sampler",
		]);

		const shaderModule = device.createShaderModule({
			label: "selective-color.wgsl",
			code: WGSL_SELECTIVE_COLOR_SHADER,
		});

		const pipeline = createFullscreenPipeline(device, {
			label: "SelectiveColorPipeline",
			module: shaderModule,
			layouts: [uniformLayout, textureLayout],
			format,
		});

		res = { pipeline, uniformLayout, textureLayout };
		deviceResourceCache.set(device, res);
	}
	return res;
}

function packAdjustment(
	data: Float32Array,
	offset: number,
	adj?: ColorAdjustment,
) {
	data[offset + 0] = (adj?.cyan ?? 0) / 100.0;
	data[offset + 1] = (adj?.magenta ?? 0) / 100.0;
	data[offset + 2] = (adj?.yellow ?? 0) / 100.0;
	data[offset + 3] = (adj?.black ?? 0) / 100.0;
}

const uniformBufferData = new Float32Array(40);

export const SelectiveColorWebGPURenderer: WebGPUNodeRenderer = async (
	args,
) => {
	const {
		ctx,
		encoder,
		pass,
		targetTexture,
		targetWidth,
		targetHeight,
		props,
	} = args;

	const op = props.virtualMedia?.operation as unknown as
		| SelectiveColorOperation
		| undefined;

	if (op?.op !== "SelectiveColor" || !props.virtualMedia?.children?.[0]) return;

	pass.end();

	const childMedia = props.virtualMedia.children[0];
	const width = targetWidth;
	const height = targetHeight;

	const { texture: tmpTex } = await drawChildToTexture(args, childMedia);

	// Prepare Uniforms
	packAdjustment(uniformBufferData, 0, op.reds);
	packAdjustment(uniformBufferData, 4, op.yellows);
	packAdjustment(uniformBufferData, 8, op.greens);
	packAdjustment(uniformBufferData, 12, op.cyans);
	packAdjustment(uniformBufferData, 16, op.blues);
	packAdjustment(uniformBufferData, 20, op.magentas);
	packAdjustment(uniformBufferData, 24, op.whites);
	packAdjustment(uniformBufferData, 28, op.neutrals);
	packAdjustment(uniformBufferData, 32, op.blacks);
	uniformBufferData[36] = op.method === "Absolute" ? 1.0 : 0.0;
	uniformBufferData[37] = 0.0;
	uniformBufferData[38] = 0.0;
	uniformBufferData[39] = 0.0;

	const { pipeline, uniformLayout, textureLayout } = getSelectiveColorResources(
		ctx.device,
		ctx.renderer.format,
	);
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
	]);

	drawTextureToTarget(args, outTex, props.opacity ?? 1);
};
