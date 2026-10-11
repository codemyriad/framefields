/// <reference types="webgpu" />
import type { SignalData, VirtualMediaData } from "@framefields/core";
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

// FIX (P1): `mapMedia` is now a first-class, typed field instead of being
// reached via `(operation as any).mapMedia`. Replace `unknown` with the
// SDK's actual virtual-media child type if one is exported.
interface DisplacementMapOp {
	op: "DisplacementMap";
	strengthX?: number;
	strengthY?: number;
	xChannel?: "Red" | "Green" | "Blue" | "Alpha" | "Luminance";
	yChannel?: "Red" | "Green" | "Blue" | "Alpha" | "Luminance";
	wrapMode?: "Clamp" | "Repeat" | "Mirror";
	opacity?: number;
	mapMedia?: VirtualMediaData | null;

	strengthXHandleId?: string | null;
	strengthYHandleId?: string | null;

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

type DisplacementSignalData = SignalData & { nodeId?: string };

// Schema bounds mirrored from displacementMapConfig.ts (`z.number().min(0).max(500)`).
// FIX (P2): kept local rather than imported to avoid pulling zod/config-builder
// into the render path; ideally these are exported as shared constants from
// displacementMapConfig.ts so the two can't drift.
const STRENGTH_MIN = 0;
const STRENGTH_MAX = 500;

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

const DISPLACEMENT_MAP_SHADER = `
struct DisplacementUniforms {
    strengthX     : f32,
    strengthY     : f32,
    xChannel      : f32, // 0=R, 1=G, 2=B, 3=A, 4=Luma
    yChannel      : f32, // 0=R, 1=G, 2=B, 3=A, 4=Luma

    wrapMode      : f32, // 0=Clamp, 1=Repeat, 2=Mirror
    hasSignalX    : f32,
    hasSignalY    : f32,
    _pad          : f32,
};

@group(0) @binding(0) var<uniform> u : DisplacementUniforms;
@group(1) @binding(0) var srcTex     : texture_2d<f32>;
@group(1) @binding(1) var srcSamp    : sampler;
@group(2) @binding(0) var mapTex     : texture_2d<f32>;
@group(2) @binding(1) var mapSamp    : sampler;
@group(3) @binding(0) var sigXTex    : texture_2d<f32>;
@group(3) @binding(1) var sigYTex    : texture_2d<f32>;
@group(3) @binding(2) var sigSamp    : sampler;
${FULLSCREEN_QUAD_WGSL}

fn extractChannel(color: vec4<f32>, channel: f32) -> f32 {
    if (channel < 0.5) { return color.r; }
    if (channel < 1.5) { return color.g; }
    if (channel < 2.5) { return color.b; }
    if (channel < 3.5) { return color.a; }
    // Luminance
    return dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
}

fn applyWrap(coord: f32, mode: f32) -> f32 {
    if (mode < 0.5) {
        // Clamp
        return clamp(coord, 0.0, 1.0);
    }
    if (mode < 1.5) {
        // Repeat
        return fract(coord);
    }
    // Mirror
    let t = fract(coord * 0.5) * 2.0;
    if (t > 1.0) {
        return 2.0 - t;
    }
    return t;
}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
    let mapColor = textureSampleLevel(mapTex, mapSamp, in.uv, 0.0);

    // Extract displacement values from chosen channels, centered around 0.5 and gated by alpha
    // NOTE (P3, not changed here): this yields a max offset of strength/2 px at
    // either extreme (peak-to-peak swing == strength). Confirm with design whether
    // "strength in pixels" should mean peak amplitude instead — see review doc.
    let dispX = (extractChannel(mapColor, u.xChannel) - 0.5) * mapColor.a;
    let dispY = (extractChannel(mapColor, u.yChannel) - 0.5) * mapColor.a;

    // Get strength, potentially from signal textures
    var sX = u.strengthX;
    var sY = u.strengthY;
    if (u.hasSignalX > 0.5) {
        sX = textureSampleLevel(sigXTex, sigSamp, in.uv, 0.0).r;
    }
    if (u.hasSignalY > 0.5) {
        sY = textureSampleLevel(sigYTex, sigSamp, in.uv, 0.0).r;
    }

    // Compute displacement in UV space (strength is in pixels, convert to UV)
    let srcDims = vec2<f32>(textureDimensions(srcTex));
    let offset = vec2<f32>(
        dispX * sX / srcDims.x,
        dispY * sY / srcDims.y,
    );

    let displaced = vec2<f32>(
        applyWrap(in.uv.x + offset.x, u.wrapMode),
        applyWrap(in.uv.y + offset.y, u.wrapMode),
    );

    return textureSampleLevel(srcTex, srcSamp, displaced, 0.0);
}
`;

// NOTE (P3, not changed): module-scoped and reused across every invocation
// (including recursive/nested displacement nodes). Currently safe because
// there is no `await` between filling this array and the synchronous
// getTemporaryBuffer() call that consumes it — if a future refactor adds an
// await in between, this becomes a cross-call data race. Consider making
// this function-scoped if that invariant ever changes.
const uniformData = new Float32Array(8);

const getDeviceLayouts = perDevice((device) => ({
	uniformLayout: fragmentBindGroupLayout(device, ["uniform"]),
	srcTextureLayout: fragmentBindGroupLayout(device, ["texture", "sampler"]),
	mapTextureLayout: fragmentBindGroupLayout(device, ["texture", "sampler"]),
	signalTextureLayout: texturesLayout(device, 2),
	pipelineCache: new Map<string, GPURenderPipeline>(),
}));

function getDisplacementResources(device: GPUDevice, format: GPUTextureFormat) {
	const layouts = getDeviceLayouts(device);

	const cacheKey = `displacement_${format}`;
	let pipeline = layouts.pipelineCache.get(cacheKey);
	if (!pipeline) {
		const module = device.createShaderModule({
			label: `displacement_${format}.wgsl`,
			code: DISPLACEMENT_MAP_SHADER,
		});

		pipeline = createFullscreenPipeline(device, {
			label: `DisplacementMapPipeline_${format}`,
			module,
			layouts: [
				layouts.uniformLayout,
				layouts.srcTextureLayout,
				layouts.mapTextureLayout,
				layouts.signalTextureLayout,
			],
			format,
		});
		layouts.pipelineCache.set(cacheKey, pipeline);
	}

	return {
		pipeline,
		uniformLayout: layouts.uniformLayout,
		srcTextureLayout: layouts.srcTextureLayout,
		mapTextureLayout: layouts.mapTextureLayout,
		signalTextureLayout: layouts.signalTextureLayout,
	};
}

function channelToFloat(
	ch: "Red" | "Green" | "Blue" | "Alpha" | "Luminance" | undefined,
): number {
	switch (ch) {
		case "Green":
			return 1;
		case "Blue":
			return 2;
		case "Alpha":
			return 3;
		case "Luminance":
			return 4;
		default:
			return 0; // Red
	}
}

function wrapModeToFloat(
	mode: "Clamp" | "Repeat" | "Mirror" | undefined,
): number {
	switch (mode) {
		case "Repeat":
			return 1;
		case "Mirror":
			return 2;
		default:
			return 0; // Clamp
	}
}

function resolveSignalField(
	op: DisplacementMapOp,
	fieldName: "strengthX" | "strengthY",
	defaultValue: number,
): { hasSignal: boolean; sd: DisplacementSignalData | null; value: number } {
	const handleIdKey = `${fieldName}HandleId` as keyof DisplacementMapOp;
	const handleId = op[handleIdKey] as string | null | undefined;
	const signalInput = handleId ? op.inputs?.[handleId] : null;

	// FIX (P2): the config's dataTypes are exactly ["Number", "Signal"] — there
	// is no "Numeric" type. Checking for it here made this branch either dead
	// code or a landmine for any output item that happened to use that string.
	// Only "Signal" should route through the per-pixel signal-texture path.
	const hasSignal = !!(
		signalInput?.connectionValid && signalInput.outputItem?.type === "Signal"
	);
	const sd = hasSignal
		? (signalInput?.outputItem?.data as DisplacementSignalData | null)
		: null;

	let value = defaultValue;
	if (!hasSignal) {
		if (
			signalInput?.connectionValid &&
			signalInput.outputItem?.type === "Number"
		) {
			// FIX (P2): clamp static Number-connection values to the same bounds
			// the field enforces when unbound, so binding a value behaves
			// consistently with typing one in directly.
			value = clamp(
				Number(signalInput.outputItem.data ?? defaultValue),
				STRENGTH_MIN,
				STRENGTH_MAX,
			);
		} else {
			value = Number(op[fieldName] ?? defaultValue);
		}
	} else if (sd && typeof sd === "object" && "offset" in sd) {
		value = Number(sd.offset ?? defaultValue);
	}

	return { hasSignal, sd, value };
}

export const DisplacementMapWebGPURenderer: WebGPUNodeRenderer = async (
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

	const op = props.virtualMedia?.operation as DisplacementMapOp | undefined;
	if (op?.op !== "DisplacementMap" || !op) return;

	const sourceMedia = props.virtualMedia.children?.[0];
	// FIX (P1): typed field access instead of `(operation as any)?.mapMedia`.
	const mapMedia = op.mapMedia;
	if (!sourceMedia || !mapMedia) return;

	// End previous pass
	pass.end();

	const width = targetWidth;
	const height = targetHeight;
	const frame = props.frame ?? 0;
	const fps = props.fps || 30;

	// Resolve signal fields
	const strengthXInfo = resolveSignalField(op, "strengthX", 50);
	const strengthYInfo = resolveSignalField(op, "strengthY", 50);

	// --- Draw source child into srcTex ---
	const { texture: srcTex } = await drawChildToTexture(args, sourceMedia, {
		props: {
			...props,
			virtualMedia: sourceMedia,
			renderId: `${props.renderId}-c0`,
		},
	});

	// FIX (P0 - CRITICAL): drawChild leaves its own last-active pass open on
	// `args.pass` (targeting srcView), matching the same "caller closes it"
	// convention this function itself uses on exit (`args.pass = finalPass`
	// below, with no `.end()`). Without this line, the next `beginFrame` call
	// (for mapClearPass) opens a second render pass on the same encoder while
	// this one is still open, which is invalid WebGPU encoder usage — it will
	// throw or push the encoder into an error state that silently drops
	// subsequent commands. This reproduces whenever `sourceMedia` renders
	// through the GPU, i.e. essentially always.
	args.pass.end();

	// --- Draw map child into mapTex ---
	const { texture: mapTex } = await drawChildToTexture(args, mapMedia, {
		props: {
			...props,
			virtualMedia: mapMedia,
			renderId: `${props.renderId}-c1`,
			excludeTextures: [...(props.excludeTextures || []), srcTex],
		},
		exclude: [srcTex],
	});

	// End active pass from child drawing
	args.pass.end();

	// --- Fetch pipeline ---
	const {
		pipeline,
		uniformLayout: uLayout,
		srcTextureLayout: srcLayout,
		mapTextureLayout: mapLayout,
		signalTextureLayout: sigLayout,
	} = getDisplacementResources(ctx.device, ctx.renderer.format);

	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);

	// --- Signal textures ---
	let sigXView: GPUTextureView;
	let sigYView: GPUTextureView;

	const { elapsedSeconds, durationSeconds } = signalTiming(props, frame, fps);

	if (strengthXInfo.hasSignal && strengthXInfo.sd) {
		sigXView = signalRegistry.getOrCreate2DTextureView(
			ctx.device,
			encoder,
			strengthXInfo.sd.nodeId ?? "displace_sx_sig",
			elapsedSeconds,
			durationSeconds,
			strengthXInfo.sd,
			width,
			height,
			props.renderId,
			frame,
			fps,
		);
	} else {
		sigXView = signalRegistry.getDummy1x1TextureView(ctx.device);
	}

	if (strengthYInfo.hasSignal && strengthYInfo.sd) {
		sigYView = signalRegistry.getOrCreate2DTextureView(
			ctx.device,
			encoder,
			strengthYInfo.sd.nodeId ?? "displace_sy_sig",
			elapsedSeconds,
			durationSeconds,
			strengthYInfo.sd,
			width,
			height,
			props.renderId,
			frame,
			fps,
		);
	} else {
		sigYView = signalRegistry.getDummy1x1TextureView(ctx.device);
	}

	// --- Fill uniforms ---
	uniformData[0] = strengthXInfo.value;
	uniformData[1] = strengthYInfo.value;
	uniformData[2] = channelToFloat(op.xChannel);
	uniformData[3] = channelToFloat(op.yChannel);
	uniformData[4] = wrapModeToFloat(op.wrapMode);
	uniformData[5] = strengthXInfo.hasSignal ? 1.0 : 0.0;
	uniformData[6] = strengthYInfo.hasSignal ? 1.0 : 0.0;
	uniformData[7] = 0.0; // padding

	const uniformBuffer = ctx.renderer.getTemporaryBuffer(uniformData);

	// --- Output texture ---
	const outTex = ctx.renderer.getTemporaryTexture(width, height, [
		srcTex,
		mapTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);

	// --- Render displacement pass ---
	drawFullscreenPass(encoder, outTex.createView(), pipeline, [
		ctx.device.createBindGroup({
			layout: uLayout,
			entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
		}),
		ctx.renderer.bindGroupCache.getBindGroup(
			ctx.device,
			srcLayout,
			srcTex,
			sampler,
		),
		ctx.renderer.bindGroupCache.getBindGroup(
			ctx.device,
			mapLayout,
			mapTex,
			sampler,
		),
		ctx.device.createBindGroup({
			layout: sigLayout,
			entries: [
				{ binding: 0, resource: sigXView },
				{ binding: 1, resource: sigYView },
				{ binding: 2, resource: sampler },
			],
		}),
	]);

	// --- Final pass: draw result to targetView ---
	drawTextureToTarget(args, outTex, op.opacity ?? 1);
};
