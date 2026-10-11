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
import { MAX_AMOUNT, MAX_RADIUS, MIN_RADIUS } from "../shared/config.js";

interface ShadowsHighlightsOp {
	op: "ShadowsHighlights";
	shadowAmount?: number;
	shadowTonalWidth?: number;
	shadowRadius?: number;
	highlightAmount?: number;
	highlightTonalWidth?: number;
	highlightRadius?: number;
	colorCorrection?: number;
	midtoneContrast?: number;

	shadowAmountHandleId?: string | null;
	shadowTonalWidthHandleId?: string | null;
	shadowRadiusHandleId?: string | null;
	highlightAmountHandleId?: string | null;
	highlightTonalWidthHandleId?: string | null;
	highlightRadiusHandleId?: string | null;
	colorCorrectionHandleId?: string | null;
	midtoneContrastHandleId?: string | null;

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

type ShadowsHighlightsSignalData = SignalData & { nodeId?: string };

const WGSL_SHADOWS_HIGHLIGHTS_UNIFORMS = `
struct ShadowsHighlightsUniforms {
	dirX                      : f32,
	dirY                      : f32,
	shadowRadius              : f32,
	highlightRadius           : f32,

	shadowAmount              : f32,
	shadowTonalWidth          : f32,
	highlightAmount           : f32,
	highlightTonalWidth       : f32,

	colorCorrection           : f32,
	midtoneContrast           : f32,
	isSecondPass              : f32,
	_pad0                     : f32,

	hasShadowAmountSig        : f32,
	hasShadowTonalWidthSig    : f32,
	hasShadowRadiusSig        : f32,
	hasHighlightAmountSig     : f32,

	hasHighlightTonalWidthSig : f32,
	hasHighlightRadiusSig     : f32,
	hasColorCorrectionSig     : f32,
	hasMidtoneContrastSig     : f32,
};

@group(0) @binding(0) var<uniform> u                : ShadowsHighlightsUniforms;
@group(1) @binding(0) var tex                       : texture_2d<f32>;
@group(1) @binding(1) var samp                      : sampler;
@group(1) @binding(2) var origTex                   : texture_2d<f32>;

@group(2) @binding(0) var shadowAmountSigTex        : texture_2d<f32>;
@group(2) @binding(1) var shadowTonalWidthSigTex    : texture_2d<f32>;
@group(2) @binding(2) var shadowRadiusSigTex        : texture_2d<f32>;
@group(2) @binding(3) var highlightAmountSigTex     : texture_2d<f32>;
@group(2) @binding(4) var highlightTonalWidthSigTex : texture_2d<f32>;
@group(2) @binding(5) var highlightRadiusSigTex     : texture_2d<f32>;
@group(2) @binding(6) var colorCorrectionSigTex     : texture_2d<f32>;
@group(2) @binding(7) var midtoneContrastSigTex     : texture_2d<f32>;
@group(2) @binding(8) var signalSamp                : sampler;
${FULLSCREEN_QUAD_WGSL}

fn get_luminance(c: vec3<f32>) -> f32 {
	return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let origCol = textureSampleLevel(origTex, samp, in.uv, 0.0);

	var sAmount = u.shadowAmount;
	if (u.hasShadowAmountSig > 0.5) {
		sAmount = textureSampleLevel(shadowAmountSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	sAmount = clamp(sAmount, 0.0, 100.0);

	var sTonalWidthVal = u.shadowTonalWidth;
	if (u.hasShadowTonalWidthSig > 0.5) {
		sTonalWidthVal = textureSampleLevel(shadowTonalWidthSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	sTonalWidthVal = clamp(sTonalWidthVal, 0.0, 100.0);

	var sRadius = u.shadowRadius;
	if (u.hasShadowRadiusSig > 0.5) {
		sRadius = textureSampleLevel(shadowRadiusSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	sRadius = clamp(sRadius, 1.0, 250.0);

	var hAmount = u.highlightAmount;
	if (u.hasHighlightAmountSig > 0.5) {
		hAmount = textureSampleLevel(highlightAmountSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	hAmount = clamp(hAmount, 0.0, 100.0);

	var hTonalWidthVal = u.highlightTonalWidth;
	if (u.hasHighlightTonalWidthSig > 0.5) {
		hTonalWidthVal = textureSampleLevel(highlightTonalWidthSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	hTonalWidthVal = clamp(hTonalWidthVal, 0.0, 100.0);

	var hRadius = u.highlightRadius;
	if (u.hasHighlightRadiusSig > 0.5) {
		hRadius = textureSampleLevel(highlightRadiusSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	hRadius = clamp(hRadius, 1.0, 250.0);

	var colCorrVal = u.colorCorrection;
	if (u.hasColorCorrectionSig > 0.5) {
		colCorrVal = textureSampleLevel(colorCorrectionSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	colCorrVal = clamp(colCorrVal, -100.0, 100.0);

	var midContrastVal = u.midtoneContrast;
	if (u.hasMidtoneContrastSig > 0.5) {
		midContrastVal = textureSampleLevel(midtoneContrastSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	midContrastVal = clamp(midContrastVal, -100.0, 100.0);

	if (sAmount <= 0.0 && hAmount <= 0.0 && abs(midContrastVal) <= 0.0) {
		if (u.isSecondPass > 0.5) {
			return origCol;
		}
		return origCol;
	}

	let dimensions = vec2<f32>(textureDimensions(tex));
	let texelSize = 1.0 / dimensions;
	let direction = vec2<f32>(u.dirX, u.dirY);

	let lumaOrig = get_luminance(origCol.rgb / max(origCol.a, 1e-5));

	// Shadow blur parameters
	let sSpatialSigma = sRadius;
	let stepScaleS = max(1.0, sSpatialSigma / 20.0);
	let effectiveSigmaS = sSpatialSigma / stepScaleS;
	let sampleRadiusS = min(i32(ceil(3.0 * effectiveSigmaS)), 64);

	// Highlight blur parameters
	let hSpatialSigma = hRadius;
	let stepScaleH = max(1.0, hSpatialSigma / 20.0);
	let effectiveSigmaH = hSpatialSigma / stepScaleH;
	let sampleRadiusH = min(i32(ceil(3.0 * effectiveSigmaH)), 64);

	if (u.isSecondPass <= 0.5) {
		// PASS 1: Horizontal blur of luminance from origTex
		// Output packed channels: R = sLumaPass1, G = hLumaPass1, B = sAlphaPass1, A = hAlphaPass1
		var sSum = 0.0;
		var sWeightSum = 0.0;
		var sAlphaSum = 0.0;
		var sKernelSum = 0.0;

		for (var i = -sampleRadiusS; i <= sampleRadiusS; i++) {
			let offset = vec2<f32>(f32(i) * stepScaleS) * direction * texelSize;
			let x = f32(i) * stepScaleS;
			let weight = exp(-0.5 * (x / sSpatialSigma) * (x / sSpatialSigma));
			let sampleCol = textureSampleLevel(tex, samp, in.uv + offset, 0.0);
			let sampleAlpha = sampleCol.a;
			let sampleLuma = get_luminance(sampleCol.rgb / max(sampleAlpha, 1e-5));
			sSum += sampleLuma * weight * sampleAlpha;
			sWeightSum += weight * sampleAlpha;
			sAlphaSum += sampleAlpha * weight;
			sKernelSum += weight;
		}

		var hSum = 0.0;
		var hWeightSum = 0.0;
		var hAlphaSum = 0.0;
		var hKernelSum = 0.0;

		for (var i = -sampleRadiusH; i <= sampleRadiusH; i++) {
			let offset = vec2<f32>(f32(i) * stepScaleH) * direction * texelSize;
			let x = f32(i) * stepScaleH;
			let weight = exp(-0.5 * (x / hSpatialSigma) * (x / hSpatialSigma));
			let sampleCol = textureSampleLevel(tex, samp, in.uv + offset, 0.0);
			let sampleAlpha = sampleCol.a;
			let sampleLuma = get_luminance(sampleCol.rgb / max(sampleAlpha, 1e-5));
			hSum += sampleLuma * weight * sampleAlpha;
			hWeightSum += weight * sampleAlpha;
			hAlphaSum += sampleAlpha * weight;
			hKernelSum += weight;
		}

		let sLumaPass1 = select(lumaOrig, sSum / sWeightSum, sWeightSum > 1e-5);
		let sAlphaPass1 = sAlphaSum / max(sKernelSum, 1e-5);
		let hLumaPass1 = select(lumaOrig, hSum / hWeightSum, hWeightSum > 1e-5);
		let hAlphaPass1 = hAlphaSum / max(hKernelSum, 1e-5);

		return vec4<f32>(sLumaPass1, hLumaPass1, sAlphaPass1, hAlphaPass1);
	}

	// PASS 2: Vertical blur of Pass 1 + Dynamic Range Recovery
	if (origCol.a < 1e-5) {
		return vec4<f32>(0.0, 0.0, 0.0, 0.0);
	}

	var sSum = 0.0;
	var sWeightSum = 0.0;

	for (var i = -sampleRadiusS; i <= sampleRadiusS; i++) {
		let offset = vec2<f32>(f32(i) * stepScaleS) * direction * texelSize;
		let x = f32(i) * stepScaleS;
		let weight = exp(-0.5 * (x / sSpatialSigma) * (x / sSpatialSigma));
		let samplePass1 = textureSampleLevel(tex, samp, in.uv + offset, 0.0);
		let sampleAlpha = samplePass1.b;
		let sampleLuma = samplePass1.r;
		sSum += sampleLuma * weight * sampleAlpha;
		sWeightSum += weight * sampleAlpha;
	}

	var hSum = 0.0;
	var hWeightSum = 0.0;

	for (var i = -sampleRadiusH; i <= sampleRadiusH; i++) {
		let offset = vec2<f32>(f32(i) * stepScaleH) * direction * texelSize;
		let x = f32(i) * stepScaleH;
		let weight = exp(-0.5 * (x / hSpatialSigma) * (x / hSpatialSigma));
		let samplePass1 = textureSampleLevel(tex, samp, in.uv + offset, 0.0);
		let sampleAlpha = samplePass1.a;
		let sampleLuma = samplePass1.g;
		hSum += sampleLuma * weight * sampleAlpha;
		hWeightSum += weight * sampleAlpha;
	}

	let sLumaBlur = select(lumaOrig, sSum / sWeightSum, sWeightSum > 1e-5);
	let hLumaBlur = select(lumaOrig, hSum / hWeightSum, hWeightSum > 1e-5);

	let alpha = origCol.a;
	let rgb = origCol.rgb / alpha;

	let sAmountNorm = sAmount / 100.0;
	let hAmountNorm = hAmount / 100.0;
	let sTonalWidth = clamp(sTonalWidthVal / 100.0, 0.01, 1.0);
	let hTonalWidth = clamp(hTonalWidthVal / 100.0, 0.01, 1.0);
	let colCorrNorm = colCorrVal / 100.0;
	let midContrastNorm = midContrastVal / 100.0;

	let sEvalLuma = 0.65 * sLumaBlur + 0.35 * lumaOrig;
	let hEvalLuma = 0.65 * hLumaBlur + 0.35 * lumaOrig;

	let wShadow = 1.0 - smoothstep(0.0, sTonalWidth, sEvalLuma);
	let wHighlight = smoothstep(1.0 - hTonalWidth, 1.0, hEvalLuma);

	let shadowBoost = sAmountNorm * wShadow * (1.0 - lumaOrig) * (1.0 - sqrt(clamp(sLumaBlur, 0.0, 1.0)));
	let highlightSuppression = hAmountNorm * wHighlight * lumaOrig * sqrt(clamp(hLumaBlur, 0.0, 1.0));

	var targetLuma = clamp(lumaOrig + shadowBoost - highlightSuppression, 0.0, 1.0);

	if (abs(midContrastNorm) > 0.001) {
		let midDiff = targetLuma - 0.5;
		let sCurve = midDiff * (1.0 - 4.0 * midDiff * midDiff);
		targetLuma = clamp(targetLuma + midContrastNorm * sCurve * 0.4, 0.0, 1.0);
	}

	var adjustedRgb = rgb * ((targetLuma + 1e-4) / (lumaOrig + 1e-4));

	let effectIntensity = sAmountNorm * wShadow + hAmountNorm * wHighlight;
	let satFactor = clamp(1.0 + colCorrNorm * effectIntensity * 1.5, 0.0, 3.0);
	adjustedRgb = mix(vec3<f32>(targetLuma), adjustedRgb, satFactor);

	let finalRgb = clamp(adjustedRgb, vec3<f32>(0.0), vec3<f32>(1.0));
	return vec4<f32>(finalRgb * alpha, alpha);
}
`;

interface DeviceShadowsHighlightsResources {
	uniformLayout: GPUBindGroupLayout;
	textureLayout: GPUBindGroupLayout;
	signalTextureLayout: GPUBindGroupLayout;
	pipeline: GPURenderPipeline;
}

const deviceResourceCache = new WeakMap<
	GPUDevice,
	DeviceShadowsHighlightsResources
>();
const shadowsHighlightsUniformData = new Float32Array(20);

function getShadowsHighlightsResources(
	device: GPUDevice,
	targetFormat: GPUTextureFormat,
): DeviceShadowsHighlightsResources {
	const cached = deviceResourceCache.get(device);
	if (cached) return cached;

	const uniformLayout = fragmentBindGroupLayout(device, ["uniform"]);

	const textureLayout = fragmentBindGroupLayout(device, [
		"texture",
		"sampler",
		"texture",
	]);

	const signalTextureLayout = texturesLayout(device, 8);

	const shaderModule = device.createShaderModule({
		code: WGSL_SHADOWS_HIGHLIGHTS_UNIFORMS,
	});

	const pipelineLayout = device.createPipelineLayout({
		bindGroupLayouts: [uniformLayout, textureLayout, signalTextureLayout],
	});

	const pipeline = createFullscreenPipeline(device, {
		module: shaderModule,
		layouts: pipelineLayout,
		format: targetFormat,
	});

	const res: DeviceShadowsHighlightsResources = {
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

	const op = props.virtualMedia?.operation as ShadowsHighlightsOp | undefined;
	if (op?.op !== "ShadowsHighlights" || !op) return;

	const childMedia = props.virtualMedia.children?.[0];
	if (!childMedia) return;

	// End incoming active pass
	pass.end();

	const frame = props.frame ?? 0;
	const fps = props.fps || 30;
	const width = targetWidth;
	const height = targetHeight;
	const nodeId =
		(props.virtualMedia?.metadata as { id?: string } | undefined)?.id ||
		(op.inputs ? Object.keys(op.inputs)[0] : null) ||
		props.renderId ||
		"shadowshighlights_node";

	const resolveField = (
		handleId: string | null | undefined,
		fallbackVal: number,
		clampMin = 0,
		clampMax = 1000,
	) =>
		resolveHandleParam<ShadowsHighlightsSignalData>(
			op.inputs,
			handleId,
			fallbackVal,
			clampMin,
			clampMax,
		);

	const shadowAmountRes = resolveField(
		op.shadowAmountHandleId,
		Number(op.shadowAmount ?? 0),
		0,
		MAX_AMOUNT,
	);
	const shadowTonalWidthRes = resolveField(
		op.shadowTonalWidthHandleId,
		Number(op.shadowTonalWidth ?? 50),
		0,
		100,
	);
	const shadowRadiusRes = resolveField(
		op.shadowRadiusHandleId,
		Number(op.shadowRadius ?? 30),
		MIN_RADIUS,
		MAX_RADIUS,
	);

	const highlightAmountRes = resolveField(
		op.highlightAmountHandleId,
		Number(op.highlightAmount ?? 0),
		0,
		MAX_AMOUNT,
	);
	const highlightTonalWidthRes = resolveField(
		op.highlightTonalWidthHandleId,
		Number(op.highlightTonalWidth ?? 50),
		0,
		100,
	);
	const highlightRadiusRes = resolveField(
		op.highlightRadiusHandleId,
		Number(op.highlightRadius ?? 30),
		MIN_RADIUS,
		MAX_RADIUS,
	);

	const colorCorrectionRes = resolveField(
		op.colorCorrectionHandleId,
		Number(op.colorCorrection ?? 0),
		-100,
		100,
	);
	const midtoneContrastRes = resolveField(
		op.midtoneContrastHandleId,
		Number(op.midtoneContrast ?? 0),
		-100,
		100,
	);

	// 1. Render child media into temporary input texture
	const { texture: tmpTex } = await drawChildToTexture(args, childMedia);

	// Check if this is a complete no-op (zero shadow/highlight recovery and zero midtone contrast)
	const isNoop =
		shadowAmountRes.val <= 0 &&
		!shadowAmountRes.hasStaticSig &&
		highlightAmountRes.val <= 0 &&
		!highlightAmountRes.hasStaticSig &&
		midtoneContrastRes.val === 0 &&
		!midtoneContrastRes.hasStaticSig;

	if (isNoop) {
		drawTextureToTarget(args, tmpTex, op.opacity ?? 1.0);
		return;
	}

	// 2. Fetch static resources and compile pipeline once
	const {
		pipeline,
		uniformLayout: uLayout,
		textureLayout: tLayout,
		signalTextureLayout: sigLayout,
	} = getShadowsHighlightsResources(ctx.device, ctx.renderer.format);

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
			sd: ShadowsHighlightsSignalData | null;
		},
		suffix: string,
	) => signalTextureView(signalContext, res, `${nodeId}_${suffix}`);

	const shadowAmountView = getSignalView(shadowAmountRes, "shadow_amount_sig");
	const shadowTonalWidthView = getSignalView(
		shadowTonalWidthRes,
		"shadow_tonal_width_sig",
	);
	const shadowRadiusView = getSignalView(shadowRadiusRes, "shadow_radius_sig");
	const highlightAmountView = getSignalView(
		highlightAmountRes,
		"highlight_amount_sig",
	);
	const highlightTonalWidthView = getSignalView(
		highlightTonalWidthRes,
		"highlight_tonal_width_sig",
	);
	const highlightRadiusView = getSignalView(
		highlightRadiusRes,
		"highlight_radius_sig",
	);
	const colorCorrectionView = getSignalView(
		colorCorrectionRes,
		"color_correction_sig",
	);
	const midtoneContrastView = getSignalView(
		midtoneContrastRes,
		"midtone_contrast_sig",
	);

	const createBindGroup = (inputTex: GPUTexture, origTexture: GPUTexture) =>
		ctx.device.createBindGroup({
			layout: tLayout,
			entries: [
				{ binding: 0, resource: inputTex.createView() },
				{ binding: 1, resource: sampler },
				{ binding: 2, resource: origTexture.createView() },
			],
		});

	// Prepare uniform data (80 bytes / 20 floats)
	shadowsHighlightsUniformData[2] = shadowRadiusRes.val;
	shadowsHighlightsUniformData[3] = highlightRadiusRes.val;
	shadowsHighlightsUniformData[4] = shadowAmountRes.val;
	shadowsHighlightsUniformData[5] = shadowTonalWidthRes.val;
	shadowsHighlightsUniformData[6] = highlightAmountRes.val;
	shadowsHighlightsUniformData[7] = highlightTonalWidthRes.val;
	shadowsHighlightsUniformData[8] = colorCorrectionRes.val;
	shadowsHighlightsUniformData[9] = midtoneContrastRes.val;
	shadowsHighlightsUniformData[10] = 0.0; // isSecondPass
	shadowsHighlightsUniformData[11] = 0.0; // _pad0
	shadowsHighlightsUniformData[12] = shadowAmountRes.hasStaticSig ? 1.0 : 0.0;
	shadowsHighlightsUniformData[13] = shadowTonalWidthRes.hasStaticSig
		? 1.0
		: 0.0;
	shadowsHighlightsUniformData[14] = shadowRadiusRes.hasStaticSig ? 1.0 : 0.0;
	shadowsHighlightsUniformData[15] = highlightAmountRes.hasStaticSig
		? 1.0
		: 0.0;
	shadowsHighlightsUniformData[16] = highlightTonalWidthRes.hasStaticSig
		? 1.0
		: 0.0;
	shadowsHighlightsUniformData[17] = highlightRadiusRes.hasStaticSig
		? 1.0
		: 0.0;
	shadowsHighlightsUniformData[18] = colorCorrectionRes.hasStaticSig
		? 1.0
		: 0.0;
	shadowsHighlightsUniformData[19] = midtoneContrastRes.hasStaticSig
		? 1.0
		: 0.0;

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
			{ binding: 0, resource: shadowAmountView },
			{ binding: 1, resource: shadowTonalWidthView },
			{ binding: 2, resource: shadowRadiusView },
			{ binding: 3, resource: highlightAmountView },
			{ binding: 4, resource: highlightTonalWidthView },
			{ binding: 5, resource: highlightRadiusView },
			{ binding: 6, resource: colorCorrectionView },
			{ binding: 7, resource: midtoneContrastView },
			{ binding: 8, resource: sampler },
		],
	});

	// Pass 1: Horizontal Gaussian convolution
	shadowsHighlightsUniformData[0] = 1.0;
	shadowsHighlightsUniformData[1] = 0.0;
	shadowsHighlightsUniformData[10] = 0.0; // isSecondPass = false
	const hBuffer = ctx.renderer.getTemporaryBuffer(shadowsHighlightsUniformData);

	drawFullscreenPass(encoder, outTex1.createView(), pipeline, [
		ctx.device.createBindGroup({
			layout: uLayout,
			entries: [{ binding: 0, resource: { buffer: hBuffer } }],
		}),
		createBindGroup(tmpTex, tmpTex),
		sigBindGroup,
	]);

	// Pass 2: Vertical Gaussian convolution + Dynamic Range Recovery
	shadowsHighlightsUniformData[0] = 0.0;
	shadowsHighlightsUniformData[1] = 1.0;
	shadowsHighlightsUniformData[10] = 1.0; // isSecondPass = true
	const vBuffer = ctx.renderer.getTemporaryBuffer(shadowsHighlightsUniformData);

	drawFullscreenPass(encoder, outTex2.createView(), pipeline, [
		ctx.device.createBindGroup({
			layout: uLayout,
			entries: [{ binding: 0, resource: { buffer: vBuffer } }],
		}),
		createBindGroup(outTex1, tmpTex),
		sigBindGroup,
	]);

	// 3. Final Pass: Draw result back to targetView using standard drawTexture
	drawTextureToTarget(args, outTex2, op.opacity ?? 1.0);
};
