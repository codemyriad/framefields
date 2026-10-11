/// <reference types="webgpu" />
import type { SignalData } from "@framefields/core";
import {
	resolveHandleParam,
	type WebGPUNodeRenderer,
} from "@framefields/node-sdk";
import {
	createFullscreenPipeline,
	drawChildToTexture,
	drawFullscreenPass,
	drawTextureToTarget,
	FULLSCREEN_QUAD_WGSL,
	fragmentBindGroupLayout,
	signalTextureView,
	signalTiming,
	texturesLayout,
} from "@framefields/node-sdk/gpu";
import { MAX_AMOUNT, MAX_RADIUS, MAX_THRESHOLD } from "../shared/config.js";

interface UnsharpMaskOp {
	op: "UnsharpMask";
	amount?: number;
	radius?: number;
	threshold?: number;
	amountHandleId?: string | null;
	radiusHandleId?: string | null;
	thresholdHandleId?: string | null;
	opacity?: number;
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

type UnsharpMaskSignalData = SignalData & { nodeId?: string };

const WGSL_UNSHARPMASK_UNIFORMS = `
struct UnsharpMaskUniforms {
	dirX           : f32,
	dirY           : f32,
	radius         : f32,
	amount         : f32,

	threshold      : f32,
	hasRadiusSig   : f32,
	hasAmountSig   : f32,
	hasThresholdSig: f32,

	isSecondPass   : f32,
	_pad0          : f32,
	_pad1          : f32,
	_pad2          : f32,
};

@group(0) @binding(0) var<uniform> u          : UnsharpMaskUniforms;
@group(1) @binding(0) var tex                 : texture_2d<f32>;
@group(1) @binding(1) var samp                : sampler;
@group(1) @binding(2) var origTex             : texture_2d<f32>;

@group(2) @binding(0) var radiusSigTex        : texture_2d<f32>;
@group(2) @binding(1) var amountSigTex        : texture_2d<f32>;
@group(2) @binding(2) var thresholdSigTex     : texture_2d<f32>;
@group(2) @binding(3) var signalSamp          : sampler;
${FULLSCREEN_QUAD_WGSL}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let dimensions = vec2<f32>(textureDimensions(tex));
	var radius = u.radius;
	if (u.hasRadiusSig > 0.5) {
		radius = textureSampleLevel(radiusSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	var amount = u.amount;
	if (u.hasAmountSig > 0.5) {
		amount = textureSampleLevel(amountSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	var threshold = u.threshold;
	if (u.hasThresholdSig > 0.5) {
		threshold = textureSampleLevel(thresholdSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}

	let origCol = textureSampleLevel(origTex, samp, in.uv, 0.0);
	if (radius <= 0.0 || amount <= 0.0) {
		if (u.isSecondPass > 0.5) {
			return origCol;
		}
		return textureSampleLevel(tex, samp, in.uv, 0.0);
	}

	let texelSize = 1.0 / dimensions;
	let spatial_sigma = radius;
	let stepScale = max(1.0, spatial_sigma / 20.0);
	let effectiveSigma = spatial_sigma / stepScale;
	let sampleRadius = min(i32(ceil(3.0 * effectiveSigma)), 64);
	let direction = vec2<f32>(u.dirX, u.dirY);

	var blurred = vec4<f32>(0.0);
	var totalWeight = 0.0;

	for (var i = -sampleRadius; i <= sampleRadius; i++) {
		let offset = vec2<f32>(f32(i) * stepScale) * direction * texelSize;
		let x = f32(i) * stepScale;
		let weight = exp(-0.5 * (x / spatial_sigma) * (x / spatial_sigma));
		blurred += textureSampleLevel(tex, samp, in.uv + offset, 0.0) * weight;
		totalWeight += weight;
	}

	let blurResult = blurred / max(totalWeight, 1e-5);

	if (u.isSecondPass <= 0.5) {
		// Pass 1: Intermediate horizontal blur
		return blurResult;
	}

	// Pass 2: Vertical blur complete -> Calculate Precision Unsharp Mask
	if (origCol.a < 1e-5) {
		return vec4<f32>(0.0, 0.0, 0.0, 0.0);
	}

	let delta = origCol.rgb - blurResult.rgb;
	let lumaDiff = abs(dot(delta, vec3<f32>(0.2126, 0.7152, 0.0722)));
	let threshNorm = (max(0.0, threshold) / 255.0) * origCol.a;

	var factor = 1.0;
	if (threshold > 0.0) {
		let ramp = (2.0 / 255.0) * origCol.a;
		factor = smoothstep(threshNorm, threshNorm + ramp, lumaDiff);
	}

	let sharpRgb = origCol.rgb + (amount / 100.0) * delta * factor;
	let clampedRgb = clamp(sharpRgb, vec3<f32>(0.0), vec3<f32>(origCol.a));
	return vec4<f32>(clampedRgb, origCol.a);
}
`;

interface DeviceUnsharpMaskResources {
	uniformLayout: GPUBindGroupLayout;
	textureLayout: GPUBindGroupLayout;
	signalTextureLayout: GPUBindGroupLayout;
	pipeline: GPURenderPipeline;
}

const deviceResourceCache = new WeakMap<
	GPUDevice,
	DeviceUnsharpMaskResources
>();
const unsharpMaskUniformData = new Float32Array(12);

function getUnsharpMaskResources(
	device: GPUDevice,
	targetFormat: GPUTextureFormat,
): DeviceUnsharpMaskResources {
	const cached = deviceResourceCache.get(device);
	if (cached) return cached;

	const uniformLayout = fragmentBindGroupLayout(device, ["uniform"]);

	const textureLayout = fragmentBindGroupLayout(device, [
		"texture",
		"sampler",
		"texture",
	]);

	const signalTextureLayout = texturesLayout(device, 3);

	const shaderModule = device.createShaderModule({
		code: WGSL_UNSHARPMASK_UNIFORMS,
	});

	const pipelineLayout = device.createPipelineLayout({
		bindGroupLayouts: [uniformLayout, textureLayout, signalTextureLayout],
	});

	const pipeline = createFullscreenPipeline(device, {
		module: shaderModule,
		layouts: pipelineLayout,
		format: targetFormat,
	});

	const res: DeviceUnsharpMaskResources = {
		uniformLayout,
		textureLayout,
		signalTextureLayout,
		pipeline,
	};
	deviceResourceCache.set(device, res);
	return res;
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

	const op = props.virtualMedia?.operation as UnsharpMaskOp | undefined;
	if (op?.op !== "UnsharpMask" || !op) return;

	const childMedia = props.virtualMedia.children?.[0];
	if (!childMedia) return;

	// End incoming active pass
	pass.end();

	const frame = props.frame ?? 0;
	const fps = props.fps || 30;
	const width = targetWidth;
	const height = targetHeight;
	const nodeId = op.inputs ? Object.keys(op.inputs)[0] : "unsharpmask_node";

	const resolveField = (
		handleId: string | null | undefined,
		fallbackVal: number,
		clampMin = 0,
		clampMax = 1000,
	) =>
		resolveHandleParam<UnsharpMaskSignalData>(
			op.inputs,
			handleId,
			fallbackVal,
			clampMin,
			clampMax,
		);

	const amountRes = resolveField(
		op.amountHandleId,
		Number(op.amount ?? 100),
		0,
		MAX_AMOUNT,
	);
	const radiusRes = resolveField(
		op.radiusHandleId,
		Number(op.radius ?? 1.5),
		0.1,
		MAX_RADIUS,
	);
	const thresholdRes = resolveField(
		op.thresholdHandleId,
		Number(op.threshold ?? 3),
		0,
		MAX_THRESHOLD,
	);

	// 1. Render child media into temporary input texture
	const { texture: tmpTex } = await drawChildToTexture(args, childMedia);

	// 2. Fetch static resources and compile pipeline once
	const {
		pipeline,
		uniformLayout: uLayout,
		textureLayout: tLayout,
		signalTextureLayout: sigLayout,
	} = getUnsharpMaskResources(ctx.device, ctx.renderer.format);

	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);

	const { elapsedSeconds, durationSeconds } = signalTiming(props, frame, fps);

	const signalContext = {
		args,
		width,
		height,
		frame,
		fps,
		elapsedSeconds,
		durationSeconds,
	};
	const getSignalView = (
		res: {
			val: number;
			hasStaticSig: boolean;
			sd: UnsharpMaskSignalData | null;
		},
		suffix: string,
	) => signalTextureView(signalContext, res, `${nodeId}_${suffix}`);

	const radiusView = getSignalView(radiusRes, "radius_sig");
	const amountView = getSignalView(amountRes, "amount_sig");
	const thresholdView = getSignalView(thresholdRes, "threshold_sig");

	const createUnsharpMaskBindGroup = (
		inputTex: GPUTexture,
		origTexture: GPUTexture,
	) =>
		ctx.device.createBindGroup({
			layout: tLayout,
			entries: [
				{ binding: 0, resource: inputTex.createView() },
				{ binding: 1, resource: sampler },
				{ binding: 2, resource: origTexture.createView() },
			],
		});

	// Prepare uniform data (48 bytes / 12 floats)
	unsharpMaskUniformData[2] = radiusRes.val;
	unsharpMaskUniformData[3] = amountRes.val;
	unsharpMaskUniformData[4] = thresholdRes.val;
	unsharpMaskUniformData[5] = radiusRes.hasStaticSig ? 1.0 : 0.0;
	unsharpMaskUniformData[6] = amountRes.hasStaticSig ? 1.0 : 0.0;
	unsharpMaskUniformData[7] = thresholdRes.hasStaticSig ? 1.0 : 0.0;
	unsharpMaskUniformData[8] = 0.0; // isSecondPass
	unsharpMaskUniformData[9] = 0.0;
	unsharpMaskUniformData[10] = 0.0;
	unsharpMaskUniformData[11] = 0.0;

	const outTex1 = ctx.renderer.getTemporaryTexture(width, height, [
		tmpTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);
	const outTex2 = ctx.renderer.getTemporaryTexture(width, height, [
		tmpTex,
		outTex1,
		targetTexture,
		...(props.excludeTextures || []),
	]);

	const sigBindGroup = ctx.device.createBindGroup({
		layout: sigLayout,
		entries: [
			{ binding: 0, resource: radiusView },
			{ binding: 1, resource: amountView },
			{ binding: 2, resource: thresholdView },
			{ binding: 3, resource: sampler },
		],
	});

	// Pass 1: Horizontal Gaussian convolution
	unsharpMaskUniformData[0] = 1.0;
	unsharpMaskUniformData[1] = 0.0;
	unsharpMaskUniformData[8] = 0.0; // isSecondPass = false
	const hBuffer = ctx.renderer.getTemporaryBuffer(unsharpMaskUniformData);

	drawFullscreenPass(encoder, outTex1.createView(), pipeline, [
		ctx.device.createBindGroup({
			layout: uLayout,
			entries: [{ binding: 0, resource: { buffer: hBuffer } }],
		}),
		createUnsharpMaskBindGroup(tmpTex, tmpTex),
		sigBindGroup,
	]);

	// Pass 2: Vertical Gaussian convolution + Precision Unsharp Mask against original tmpTex
	unsharpMaskUniformData[0] = 0.0;
	unsharpMaskUniformData[1] = 1.0;
	unsharpMaskUniformData[8] = 1.0; // isSecondPass = true
	const vBuffer = ctx.renderer.getTemporaryBuffer(unsharpMaskUniformData);

	drawFullscreenPass(encoder, outTex2.createView(), pipeline, [
		ctx.device.createBindGroup({
			layout: uLayout,
			entries: [{ binding: 0, resource: { buffer: vBuffer } }],
		}),
		createUnsharpMaskBindGroup(outTex1, tmpTex),
		sigBindGroup,
	]);

	// 3. Final Pass: Draw result back to targetView using standard drawTexture for blending/opacity
	drawTextureToTarget(args, outTex2, op.opacity ?? 1.0);
};
