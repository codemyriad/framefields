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
} from "@framefields/node-sdk/gpu";
import { lutStore, signalRegistry } from "@framefields/webgpu-renderers";

export const LUT_3D_SHADER = `
struct LutUniforms {
	intensity      : f32,
	lutSize        : f32,
	hasSignal      : f32,
	pad            : f32,
};

@group(0) @binding(0) var<uniform> u : LutUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;
@group(2) @binding(0) var lutTex     : texture_3d<f32>;
@group(2) @binding(1) var lutSamp    : sampler;
@group(3) @binding(0) var signalTex  : texture_2d<f32>;
@group(3) @binding(1) var signalSamp : sampler;
${FULLSCREEN_QUAD_WGSL}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let inputColor = textureSampleLevel(tex, samp, in.uv, 0.0);
	
	// Coordinate adjustment to sample from texel centers
	let uvw = inputColor.rgb * ((u.lutSize - 1.0) / u.lutSize) + vec3<f32>(0.5 / u.lutSize);
	let mappedColor = textureSampleLevel(lutTex, lutSamp, uvw, 0.0);
	
	var activeIntensity = u.intensity;
	if (u.hasSignal > 0.5) {
		activeIntensity = textureSampleLevel(signalTex, signalSamp, in.uv, 0.0).r;
	}
	
	let finalRgb = inputColor.rgb + activeIntensity  * (mappedColor.rgb - inputColor.rgb);
	return vec4<f32>(finalRgb * inputColor.a, inputColor.a);
}
`;

export const LUT_1D_SHADER = `
struct LutUniforms {
	intensity      : f32,
	lutSize        : f32,
	hasSignal      : f32,
	pad            : f32,
};

@group(0) @binding(0) var<uniform> u : LutUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;
@group(2) @binding(0) var lutTex     : texture_1d<f32>;
@group(2) @binding(1) var lutSamp    : sampler;
@group(3) @binding(0) var signalTex  : texture_2d<f32>;
@group(3) @binding(1) var signalSamp : sampler;
${FULLSCREEN_QUAD_WGSL}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let inputColor = textureSampleLevel(tex, samp, in.uv, 0.0);
	
	let rCoord = inputColor.r * ((u.lutSize - 1.0) / u.lutSize) + (0.5 / u.lutSize);
	let gCoord = inputColor.g * ((u.lutSize - 1.0) / u.lutSize) + (0.5 / u.lutSize);
	let bCoord = inputColor.b * ((u.lutSize - 1.0) / u.lutSize) + (0.5 / u.lutSize);
	
	let mappedR = textureSampleLevel(lutTex, lutSamp, rCoord, 0.0).r;
	let mappedG = textureSampleLevel(lutTex, lutSamp, gCoord, 0.0).g;
	let mappedB = textureSampleLevel(lutTex, lutSamp, bCoord, 0.0).b;
	
	var activeIntensity = u.intensity;
	if (u.hasSignal > 0.5) {
		activeIntensity = textureSampleLevel(signalTex, signalSamp, in.uv, 0.0).r;
	}
	
	let finalRgb = inputColor.rgb + activeIntensity  * (vec3<f32>(mappedR, mappedG, mappedB) - inputColor.rgb);
	return vec4<f32>(finalRgb * inputColor.a, inputColor.a);
}
`;

const lutUniformData = new Float32Array(4);

const getDeviceLayouts = perDevice((device) => ({
	lutUniformLayout: fragmentBindGroupLayout(device, ["uniform"]),
	singleTextureLayout: fragmentBindGroupLayout(device, ["texture", "sampler"]),
	lutTextureLayout3D: fragmentBindGroupLayout(device, [
		{ texture: { viewDimension: "3d", sampleType: "float" } },
		"sampler",
	]),
	lutTextureLayout1D: fragmentBindGroupLayout(device, [
		{ texture: { viewDimension: "1d", sampleType: "float" } },
		"sampler",
	]),
	signalTextureLayout: fragmentBindGroupLayout(device, ["texture", "sampler"]),
	pipelineCache: new Map<string, GPURenderPipeline>(),
}));

function getLutResources(
	device: GPUDevice,
	format: GPUTextureFormat,
	type: "1D" | "3D",
) {
	const layouts = getDeviceLayouts(device);
	const cacheKey = `lut_${type}_${format}`;
	let pipeline = layouts.pipelineCache.get(cacheKey);
	if (!pipeline) {
		const shaderCode = type === "3D" ? LUT_3D_SHADER : LUT_1D_SHADER;
		const lutModule = device.createShaderModule({
			label: `lut_${type.toLowerCase()}_${format}.wgsl`,
			code: shaderCode,
		});
		const currentLutLayout =
			type === "3D" ? layouts.lutTextureLayout3D : layouts.lutTextureLayout1D;
		pipeline = createFullscreenPipeline(device, {
			label: `LutPipeline_${type}_${format}`,
			module: lutModule,
			layouts: [
				layouts.lutUniformLayout,
				layouts.singleTextureLayout,
				currentLutLayout,
				layouts.signalTextureLayout,
			],
			format,
		});
		layouts.pipelineCache.set(cacheKey, pipeline);
	}

	return {
		lutPipeline: pipeline,
		lutUniformLayout: layouts.lutUniformLayout,
		singleTextureLayout: layouts.singleTextureLayout,
		lutTextureLayout:
			type === "3D" ? layouts.lutTextureLayout3D : layouts.lutTextureLayout1D,
		signalTextureLayout: layouts.signalTextureLayout,
	};
}

export const LutWebGPURenderer: WebGPUNodeRenderer = async (args) => {
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
	const op = props.virtualMedia?.operation as any;
	if (op?.op !== "ApplyLUT" || !op) return;
	const childMedia = props.virtualMedia.children?.[0];
	if (!childMedia) return;

	const lutUrl = op.lutUrl;
	if (!lutUrl) return;

	// End the active pass
	pass.end();

	const width = targetWidth;
	const height = targetHeight;

	// Resolve the intensity signal
	const intensityInput = op.intensityHandleId
		? op.inputs?.[op.intensityHandleId]
		: null;
	const hasSignal = !!(
		intensityInput?.connectionValid &&
		intensityInput.outputItem?.type === "Signal"
	);
	const sd = hasSignal ? intensityInput?.outputItem?.data : null;

	let intensity = 1.0;
	let hasStaticSignal = false;

	if (!hasSignal) {
		if (
			intensityInput?.connectionValid &&
			intensityInput.outputItem?.type === "Number"
		) {
			intensity = Number(intensityInput.outputItem.data ?? 1.0);
		} else {
			intensity = Number(op.intensity ?? 1.0);
		}
	} else if (sd && typeof sd === "object") {
		intensity = Number((sd as any).offset ?? 1.0);
		hasStaticSignal = true;
	}
	intensity = Math.max(0, Math.min(10, intensity));

	// Load the GPU LUT resource without blocking the rendering pipeline.
	let lut = lutStore.get(lutUrl, ctx.device);
	if (!lut) {
		const isHeadless =
			props.isHeadless ??
			(typeof window === "undefined" ||
				(globalThis as any).__IS_HEADLESS_RENDERER__);

		if (isHeadless || props.forceWait) {
			try {
				lut = await lutStore.getOrLoad(lutUrl, ctx.device);
			} catch (err) {
				if (
					!(err instanceof DOMException && err.name === "AbortError") &&
					!(
						err &&
						typeof err === "object" &&
						"name" in err &&
						err.name === "AbortError"
					)
				) {
					console.error("[ApplyLUT] Failed to load LUT synchronously:", err);
				}
			}
		} else {
			// Trigger background load
			lutStore.getOrLoad(lutUrl, ctx.device).catch((err) => {
				if (
					(err instanceof DOMException && err.name === "AbortError") ||
					(err &&
						typeof err === "object" &&
						"name" in err &&
						err.name === "AbortError")
				) {
					return;
				}
				console.error("[ApplyLUT] Failed to load LUT in background:", err);
			});
		}

		// Fallback to any matching cached LUT for the same node
		if (!lut && lutUrl.startsWith("runtime://lut/")) {
			const prefix = lutUrl.substring(0, lutUrl.lastIndexOf("-") + 1);
			if (prefix) {
				lut = lutStore.getAnyMatching(prefix, ctx.device);
			}
		}
	}

	const { texture: tmpTex } = await drawChildToTexture(args, childMedia);

	// If no LUT is ready (and no fallback is available), draw the child frame directly as a passthrough
	if (!lut) {
		drawTextureToTarget(args, tmpTex, op.opacity ?? 1);
		return;
	}

	// Fetch pipeline and layouts
	const {
		lutPipeline: pipeline,
		lutUniformLayout: uLayout,
		singleTextureLayout: tLayout,
		lutTextureLayout: lLayout,
		signalTextureLayout: sigLayout,
	} = getLutResources(ctx.device, ctx.renderer.format, lut.type);

	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);

	// Setup intermediate output texture
	const outTex = ctx.renderer.getTemporaryTexture(width, height, [
		tmpTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);

	// Fetch dynamic intensity signal texture view
	let signalView: GPUTextureView;
	if (hasStaticSignal) {
		const { elapsedSeconds, durationSeconds } = signalTiming(props, frame, fps);
		signalView = signalRegistry.getOrCreate2DTextureView(
			ctx.device,
			encoder,
			sd.nodeId ?? "lut_intensity_sig",
			elapsedSeconds,
			durationSeconds,
			sd,
			width,
			height,
			props.renderId,
			frame,
			fps,
		);
	} else {
		signalView = signalRegistry.getDummy1x1TextureView(ctx.device);
	}

	// Prepare uniforms
	lutUniformData[0] = intensity;
	lutUniformData[1] = lut.size;
	lutUniformData[2] = hasStaticSignal ? 1 : 0;
	lutUniformData[3] = 0; // padding

	const uniformBuffer = ctx.renderer.getTemporaryBuffer(lutUniformData);

	drawFullscreenPass(encoder, outTex.createView(), pipeline!, [
		ctx.device.createBindGroup({
			layout: uLayout!,
			entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
		}),
		ctx.renderer.bindGroupCache.getBindGroup(
			ctx.device,
			tLayout!,
			tmpTex,
			sampler,
		),
		ctx.device.createBindGroup({
			layout: lLayout!,
			entries: [
				{ binding: 0, resource: lut.texture.createView() },
				{ binding: 1, resource: sampler },
			],
		}),
		ctx.device.createBindGroup({
			layout: sigLayout!,
			entries: [
				{ binding: 0, resource: signalView },
				{ binding: 1, resource: sampler },
			],
		}),
	]);

	// Blending/opacity draw to the targetView
	drawTextureToTarget(args, outTex, op.opacity ?? 1);
};
