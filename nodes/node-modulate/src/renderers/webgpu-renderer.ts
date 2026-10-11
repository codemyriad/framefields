/// <reference types="webgpu" />
import type { WebGPUNodeRenderer } from "@framefields/node-sdk";
import {
	createFullscreenPipeline,
	drawChildToTexture,
	drawFullscreenPass,
	drawTextureToTarget,
	FULLSCREEN_QUAD_WGSL,
	fragmentBindGroupLayout,
	perDevice,
	signalTiming,
	texturesLayout,
} from "@framefields/node-sdk/gpu";
import { signalRegistry } from "@framefields/webgpu-renderers";

interface ModulateOp {
	op?: string;
	hueHandleId?: string | null;
	brightnessHandleId?: string | null;
	contrastHandleId?: string | null;
	exposureHandleId?: string | null;
	saturationHandleId?: string | null;
	sepiaHandleId?: string | null;
	hue?: number;
	brightness?: number;
	contrast?: number;
	exposure?: number;
	saturation?: number;
	sepia?: number;
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

interface FieldSignalInfo {
	hasSignal: boolean;
	sd: any;
	value: number;
}

function resolveVal(
	val: unknown,
	fallback: number,
	frame: number,
	fps: number,
): number {
	if (val === undefined || val === null) return fallback;
	if (typeof val === "object" && val !== null) {
		if (
			"get" in (val as Record<string, unknown>) &&
			typeof (val as { get: unknown }).get === "function"
		) {
			try {
				const res = (val as { get: (ctx?: unknown) => unknown }).get({
					frame,
					fps,
					time: fps > 0 ? frame / fps : 0,
					duration: 0,
					durationMs: 0,
					progress: 0,
					deltaTime: fps > 0 ? 1 / fps : 0,
				});
				const n = Number(res);
				if (Number.isFinite(n)) return n;
			} catch {
				// fallback
			}
		}
		if ("value" in (val as Record<string, unknown>)) {
			const n = Number((val as { value: unknown }).value);
			if (Number.isFinite(n)) return n;
		}
	} else if (typeof val === "function") {
		try {
			const res = (val as (f: number, fps: number) => unknown)(frame, fps);
			const n = Number(res);
			if (Number.isFinite(n)) return n;
		} catch {
			// fallback
		}
	}
	const n = Number(val);
	return Number.isFinite(n) ? n : fallback;
}

function resolveModulateField(
	op: ModulateOp,
	fieldName:
		| "hue"
		| "brightness"
		| "contrast"
		| "exposure"
		| "saturation"
		| "sepia",
	defaultValue: number,
	minVal: number,
	maxVal: number,
	frame = 0,
	fps = 30,
): FieldSignalInfo {
	const handleIdKey = `${fieldName}HandleId` as keyof ModulateOp;
	const handleId = op[handleIdKey] as string | null | undefined;
	const signalInput = handleId ? op.inputs?.[handleId] : null;

	const hasSignal = !!(
		signalInput?.connectionValid && signalInput.outputItem?.type === "Signal"
	);
	const sd = hasSignal ? signalInput?.outputItem?.data : null;

	let value = defaultValue;
	if (!hasSignal) {
		if (
			signalInput?.connectionValid &&
			signalInput.outputItem?.type === "Number"
		) {
			value = Number(signalInput.outputItem.data ?? defaultValue);
		} else {
			const raw =
				op[fieldName] ?? (op as Record<string, unknown>).signals?.[fieldName];
			value = resolveVal(raw, defaultValue, frame, fps);
		}
	} else if (sd && typeof sd === "object" && "offset" in sd) {
		value = Number(sd.offset ?? defaultValue);
	}

	value = Math.max(minVal, Math.min(maxVal, value));
	return { hasSignal, sd, value };
}

export const STATIC_MODULATE_SHADER = `
struct ModulateUniforms {
	hue                  : f32,
	brightness           : f32,
	contrast             : f32,
	exposure             : f32,
	saturation           : f32,
	sepia                : f32,

	hasSignal_hue        : f32,
	hasSignal_brightness : f32,
	hasSignal_contrast   : f32,
	hasSignal_exposure   : f32,
	hasSignal_saturation : f32,
	hasSignal_sepia      : f32,
};

@group(0) @binding(0) var<uniform> u : ModulateUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;

@group(2) @binding(0) var hueTex        : texture_2d<f32>;
@group(2) @binding(1) var brightnessTex : texture_2d<f32>;
@group(2) @binding(2) var contrastTex   : texture_2d<f32>;
@group(2) @binding(3) var exposureTex   : texture_2d<f32>;
@group(2) @binding(4) var saturationTex : texture_2d<f32>;
@group(2) @binding(5) var sepiaTex      : texture_2d<f32>;
@group(2) @binding(6) var sigSamp       : sampler;
${FULLSCREEN_QUAD_WGSL}

fn apply_exposure(c: vec4<f32>, exposure: f32) -> vec4<f32> {
	let scale = pow(2.0, exposure);
	var unpremult = c;
	if (unpremult.a > 0.0) { unpremult = vec4<f32>(unpremult.rgb / unpremult.a, unpremult.a); }
	let r = scale * unpremult.r;
	let g = scale * unpremult.g;
	let b = scale * unpremult.b;
	let outAlpha = clamp(unpremult.a, 0.0, 1.0);
	let outRgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)) * outAlpha;
	return vec4<f32>(outRgb, outAlpha);
}

fn apply_hue(c: vec4<f32>, degrees: f32) -> vec4<f32> {
	let rad = (degrees * 3.1415926535) / 180.0;
	let cosVal = cos(rad);
	let sinVal = sin(rad);

	let row0 = vec4<f32>(0.213 + cosVal * 0.787 - sinVal * 0.213, 0.715 - cosVal * 0.715 - sinVal * 0.715, 0.072 - cosVal * 0.072 + sinVal * 0.928, 0.0);
	let row1 = vec4<f32>(0.213 - cosVal * 0.213 + sinVal * 0.143, 0.715 + cosVal * 0.285 + sinVal * 0.14,  0.072 - cosVal * 0.072 - sinVal * 0.283, 0.0);
	let row2 = vec4<f32>(0.213 - cosVal * 0.213 - sinVal * 0.787, 0.715 - cosVal * 0.715 + sinVal * 0.715, 0.072 + cosVal * 0.928 + sinVal * 0.072, 0.0);
	
	var unpremult = c;
	if (unpremult.a > 0.0) { unpremult = vec4<f32>(unpremult.rgb / unpremult.a, unpremult.a); }
	let r = dot(row0, unpremult);
	let g = dot(row1, unpremult);
	let b = dot(row2, unpremult);
	let outAlpha = clamp(unpremult.a, 0.0, 1.0);
	let outRgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)) * outAlpha;
	return vec4<f32>(outRgb, outAlpha);
}

fn apply_saturation(c: vec4<f32>, sat: f32) -> vec4<f32> {
	let row0 = vec4<f32>(0.213 + sat * 0.787, 0.715 * (1.0 - sat), 0.072 * (1.0 - sat), 0.0);
	let row1 = vec4<f32>(0.213 * (1.0 - sat), 0.715 + sat * 0.285, 0.072 * (1.0 - sat), 0.0);
	let row2 = vec4<f32>(0.213 * (1.0 - sat), 0.715 * (1.0 - sat), 0.072 + sat * 0.928, 0.0);

	var unpremult = c;
	if (unpremult.a > 0.0) { unpremult = vec4<f32>(unpremult.rgb / unpremult.a, unpremult.a); }
	let r = dot(row0, unpremult);
	let g = dot(row1, unpremult);
	let b = dot(row2, unpremult);
	let outAlpha = clamp(unpremult.a, 0.0, 1.0);
	let outRgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)) * outAlpha;
	return vec4<f32>(outRgb, outAlpha);
}

fn apply_contrast(c: vec4<f32>, contrast: f32) -> vec4<f32> {
	let offset = 0.5 * (1.0 - contrast);
	
	var unpremult = c;
	if (unpremult.a > 0.0) { unpremult = vec4<f32>(unpremult.rgb / unpremult.a, unpremult.a); }
	let r = contrast * unpremult.r + offset;
	let g = contrast * unpremult.g + offset;
	let b = contrast * unpremult.b + offset;
	let outAlpha = clamp(unpremult.a, 0.0, 1.0);
	let outRgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)) * outAlpha;
	return vec4<f32>(outRgb, outAlpha);
}

fn apply_brightness(c: vec4<f32>, bright: f32) -> vec4<f32> {
	var unpremult = c;
	if (unpremult.a > 0.0) { unpremult = vec4<f32>(unpremult.rgb / unpremult.a, unpremult.a); }
	let r = bright * unpremult.r;
	let g = bright * unpremult.g;
	let b = bright * unpremult.b;
	let outAlpha = clamp(unpremult.a, 0.0, 1.0);
	let outRgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)) * outAlpha;
	return vec4<f32>(outRgb, outAlpha);
}

fn apply_sepia(c: vec4<f32>, s: f32) -> vec4<f32> {
	let row0 = vec4<f32>(1.0 - 0.607 * s, 0.769 * s,       0.189 * s,       0.0);
	let row1 = vec4<f32>(0.349 * s,       1.0 - 0.314 * s, 0.168 * s,       0.0);
	let row2 = vec4<f32>(0.272 * s,       0.534 * s,       1.0 - 0.869 * s, 0.0);

	var unpremult = c;
	if (unpremult.a > 0.0) { unpremult = vec4<f32>(unpremult.rgb / unpremult.a, unpremult.a); }
	let r = dot(row0, unpremult);
	let g = dot(row1, unpremult);
	let b = dot(row2, unpremult);
	let outAlpha = clamp(unpremult.a, 0.0, 1.0);
	let outRgb = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)) * outAlpha;
	return vec4<f32>(outRgb, outAlpha);
}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let original = textureSample(tex, samp, in.uv);
	var color = original;

	// 1. Exposure
	var exposure = u.exposure;
	if (u.hasSignal_exposure > 0.5) {
		exposure = textureSampleLevel(exposureTex, sigSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	color = apply_exposure(color, exposure);

	// 2. Hue
	var hue = u.hue;
	if (u.hasSignal_hue > 0.5) {
		hue = textureSampleLevel(hueTex, sigSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	color = apply_hue(color, hue);

	// 3. Saturation
	var saturation = u.saturation;
	if (u.hasSignal_saturation > 0.5) {
		saturation = textureSampleLevel(saturationTex, sigSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	color = apply_saturation(color, saturation);

	// 4. Contrast
	var contrast = u.contrast;
	if (u.hasSignal_contrast > 0.5) {
		contrast = textureSampleLevel(contrastTex, sigSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	color = apply_contrast(color, contrast);

	// 5. Brightness
	var brightness = u.brightness;
	if (u.hasSignal_brightness > 0.5) {
		brightness = textureSampleLevel(brightnessTex, sigSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	color = apply_brightness(color, brightness);

	// 6. Sepia
	var sepia = u.sepia;
	if (u.hasSignal_sepia > 0.5) {
		sepia = textureSampleLevel(sepiaTex, sigSamp, vec2<f32>(0.5, 0.5), 0.0).r;
	}
	color = apply_sepia(color, sepia);

	return color;
}
`;

const modulateData = new Float32Array(12);

const getDeviceLayouts = perDevice((device) => ({
	modulateUniformLayout: fragmentBindGroupLayout(device, ["uniform"]),
	singleTextureLayout: fragmentBindGroupLayout(device, ["texture", "sampler"]),
	modulateSignalsLayout: texturesLayout(device, 6),
	pipelineCache: new Map<string, GPURenderPipeline>(),
}));

function getModulateResources(device: GPUDevice, format: GPUTextureFormat) {
	const layouts = getDeviceLayouts(device);

	const cacheKey = `modulate_${format}`;
	let pipeline = layouts.pipelineCache.get(cacheKey);
	if (!pipeline) {
		const modulateModule = device.createShaderModule({
			label: `modulate_static_${format}.wgsl`,
			code: STATIC_MODULATE_SHADER,
		});

		pipeline = createFullscreenPipeline(device, {
			label: `ModulatePipeline_static_${format}`,
			module: modulateModule,
			layouts: [
				layouts.modulateUniformLayout,
				layouts.singleTextureLayout,
				layouts.modulateSignalsLayout,
			],
			format,
		});
		layouts.pipelineCache.set(cacheKey, pipeline);
	}

	return {
		modulatePipeline: pipeline,
		modulateUniformLayout: layouts.modulateUniformLayout,
		singleTextureLayout: layouts.singleTextureLayout,
		modulateSignalsLayout: layouts.modulateSignalsLayout,
	};
}

export const ModulateWebGPURenderer: WebGPUNodeRenderer = async (args) => {
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
	const compositionFrame = props.compositionFrame ?? frame;
	const fps = props.fps || 30;
	const op = props.virtualMedia?.operation as ModulateOp | undefined;
	if (op?.op !== "Modulate" || !op) return;

	const childMedia = props.virtualMedia.children?.[0];
	if (!childMedia) return;

	pass.end();

	const hueInfo = resolveModulateField(
		op,
		"hue",
		0,
		0,
		360,
		compositionFrame,
		fps,
	);
	const brightnessInfo = resolveModulateField(
		op,
		"brightness",
		1,
		0,
		2,
		compositionFrame,
		fps,
	);
	const contrastInfo = resolveModulateField(
		op,
		"contrast",
		1,
		0,
		2,
		compositionFrame,
		fps,
	);
	const exposureInfo = resolveModulateField(
		op,
		"exposure",
		0,
		-2,
		2,
		compositionFrame,
		fps,
	);
	const saturationInfo = resolveModulateField(
		op,
		"saturation",
		1,
		0,
		2,
		compositionFrame,
		fps,
	);
	const sepiaInfo = resolveModulateField(
		op,
		"sepia",
		0,
		0,
		1,
		compositionFrame,
		fps,
	);

	const width = targetWidth;
	const height = targetHeight;

	const { texture: tmpTex } = await drawChildToTexture(args, childMedia);

	const {
		modulatePipeline: pipeline,
		modulateUniformLayout: uLayout,
		singleTextureLayout: tLayout,
		modulateSignalsLayout: sigLayout,
	} = getModulateResources(ctx.device, ctx.renderer.format);

	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);

	const outTex = ctx.renderer.getTemporaryTexture(width, height, [
		tmpTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);

	// Pack uniforms:
	// indices 0-5: base values for hue, brightness, contrast, exposure, saturation, sepia (Float32)
	// indices 6-11: active flags for hue, brightness, contrast, exposure, saturation, sepia (Uint32, as Float bitcast or float value)
	modulateData[0] = hueInfo.value;
	modulateData[1] = brightnessInfo.value;
	modulateData[2] = contrastInfo.value;
	modulateData[3] = exposureInfo.value;
	modulateData[4] = saturationInfo.value;
	modulateData[5] = sepiaInfo.value;

	const fields = [
		{ name: "hue", info: hueInfo, idx: 6 },
		{ name: "brightness", info: brightnessInfo, idx: 7 },
		{ name: "contrast", info: contrastInfo, idx: 8 },
		{ name: "exposure", info: exposureInfo, idx: 9 },
		{ name: "saturation", info: saturationInfo, idx: 10 },
		{ name: "sepia", info: sepiaInfo, idx: 11 },
	];

	const signalViews: GPUTextureView[] = [];

	const { elapsedSeconds, durationSeconds } = signalTiming(props, frame, fps);

	for (const f of fields) {
		const isGenerator = !!(f.info.hasSignal && f.info.sd);
		modulateData[f.idx] = isGenerator ? 1 : 0;

		if (isGenerator) {
			const view = signalRegistry.getOrCreate2DTextureView(
				ctx.device,
				encoder,
				f.info.sd.nodeId ?? `modulate_${f.name}_sig`,
				elapsedSeconds,
				durationSeconds,
				f.info.sd,
				width,
				height,
				props.renderId,
				frame,
				fps,
			);
			signalViews.push(view);
		} else {
			signalViews.push(signalRegistry.getDummy1x1TextureView(ctx.device));
		}
	}

	const buffer = ctx.renderer.getTemporaryBuffer(modulateData);

	drawFullscreenPass(encoder, outTex.createView(), pipeline!, [
		ctx.device.createBindGroup({
			layout: uLayout!,
			entries: [{ binding: 0, resource: { buffer } }],
		}),
		ctx.renderer.bindGroupCache.getBindGroup(
			ctx.device,
			tLayout!,
			tmpTex,
			sampler,
		),
		ctx.device.createBindGroup({
			layout: sigLayout!,
			entries: [
				{ binding: 0, resource: signalViews[0] }, // hue
				{ binding: 1, resource: signalViews[1] }, // brightness
				{ binding: 2, resource: signalViews[2] }, // contrast
				{ binding: 3, resource: signalViews[3] }, // exposure
				{ binding: 4, resource: signalViews[4] }, // saturation
				{ binding: 5, resource: signalViews[5] }, // sepia
				{ binding: 6, resource: sampler },
			],
		}),
	]);

	drawTextureToTarget(args, outTex, op.opacity ?? 1);
};
