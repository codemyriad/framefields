/// <reference types="webgpu" />
import type { SignalData } from "@framefields/core";
import {
	unwrapFrameParam,
	type WebGPUNodeRenderer,
} from "@framefields/node-sdk";
import {
	createFullscreenPipeline,
	drawChildToTexture,
	drawFullscreenPass,
	drawTextureToTarget,
	FULLSCREEN_QUAD_WGSL,
	fragmentBindGroupLayout,
	perDevice,
	signalTextureView,
	signalTiming,
	texturesLayout,
} from "@framefields/node-sdk/gpu";

interface TileOffsetOp {
	op: "TileOffset";
	offsetX?: number;
	offsetY?: number;
	wrap?: boolean;
	edgeMode?: "wrap" | "clamp" | "transparent" | "mirror";
	opacity?: number;
	offsetXHandleId?: string | null;
	offsetYHandleId?: string | null;
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

type TileOffsetSignalData = SignalData & { nodeId?: string };

const TILE_OFFSET_SHADER = `
struct TileOffsetUniforms {
    offsetX        : f32,
    offsetY        : f32,
    hasOffsetXSig  : f32,
    hasOffsetYSig  : f32,

    layerWidth     : f32,
    layerHeight    : f32,
    edgeMode       : f32,
    pad            : f32,
};

@group(0) @binding(0) var<uniform> u : TileOffsetUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;

@group(2) @binding(0) var offsetXSigTex : texture_2d<f32>;
@group(2) @binding(1) var offsetYSigTex : texture_2d<f32>;
@group(2) @binding(2) var signalSamp    : sampler;
${FULLSCREEN_QUAD_WGSL}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
    var offX = u.offsetX;
    if (u.hasOffsetXSig > 0.5) {
        let sigX = textureSampleLevel(offsetXSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
        offX = offX + sigX;
    }

    var offY = u.offsetY;
    if (u.hasOffsetYSig > 0.5) {
        let sigY = textureSampleLevel(offsetYSigTex, signalSamp, vec2<f32>(0.5, 0.5), 0.0).r;
        offY = offY + sigY;
    }

    let normOffset = vec2<f32>(
        offX / max(1.0, u.layerWidth),
        offY / max(1.0, u.layerHeight)
    );
    var uv = in.uv - normOffset;

    // Edge Modes:
    // 0.0 = wrap (seamless modulo)
    // 1.0 = clamp (edge smear)
    // 2.0 = transparent (clear bounds)
    // 3.0 = mirror (ping-pong reflection)
    if (u.edgeMode < 0.5) {
        uv = fract(uv);
    } else if (u.edgeMode < 1.5) {
        uv = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    } else if (u.edgeMode < 2.5) {
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            return vec4<f32>(0.0);
        }
    } else {
        uv = 1.0 - abs(fract(uv * 0.5) * 2.0 - 1.0);
    }

    let color = textureSampleLevel(tex, samp, uv, 0.0);
    return color;
}
`;

const uniformData = new Float32Array(8);

const getDeviceLayouts = perDevice((device) => ({
	uniformLayout: fragmentBindGroupLayout(device, ["uniform"]),
	singleTextureLayout: fragmentBindGroupLayout(device, ["texture", "sampler"]),
	signalTextureLayout: texturesLayout(device, 2),
	pipelineCache: new Map<string, GPURenderPipeline>(),
}));

function getTileOffsetResources(device: GPUDevice, format: GPUTextureFormat) {
	const layouts = getDeviceLayouts(device);
	const cacheKey = `tile_offset_${format}`;
	let pipeline = layouts.pipelineCache.get(cacheKey);

	if (!pipeline) {
		const module = device.createShaderModule({
			label: `tile_offset_${format}.wgsl`,
			code: TILE_OFFSET_SHADER,
		});

		pipeline = createFullscreenPipeline(device, {
			label: `TileOffsetPipeline_${format}`,
			module,
			layouts: [
				layouts.uniformLayout,
				layouts.singleTextureLayout,
				layouts.signalTextureLayout,
			],
			format,
		});
		layouts.pipelineCache.set(cacheKey, pipeline);
	}

	return {
		pipeline,
		uniformLayout: layouts.uniformLayout,
		singleTextureLayout: layouts.singleTextureLayout,
		signalTextureLayout: layouts.signalTextureLayout,
	};
}

export const TileOffsetWebGPURenderer: WebGPUNodeRenderer = async (args) => {
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
	const op = props.virtualMedia?.operation as TileOffsetOp | undefined;
	if (op?.op !== "TileOffset" || !op) return;

	const childMedia = props.virtualMedia?.children?.[0];
	if (!childMedia) return;

	// End active pass before rendering child
	pass.end();

	const width = targetWidth;
	const height = targetHeight;
	const nodeId = op.inputs ? Object.keys(op.inputs)[0] : "tile_offset_node";

	// 1. Resolve Bindable Parameters
	const resolveBindable = (
		configKey: "offsetX" | "offsetY",
		defaultValue: number,
	) => {
		const handleIdKey = `${configKey}HandleId` as keyof TileOffsetOp;
		const handleId = op[handleIdKey] as string | null | undefined;
		const input = handleId ? op.inputs?.[handleId] : null;

		const hasSignal = !!(
			input?.connectionValid &&
			(input.outputItem?.type === "Signal" ||
				input.outputItem?.type === "Numeric")
		);

		const sd =
			hasSignal && input?.outputItem?.data
				? (input.outputItem.data as TileOffsetSignalData)
				: null;

		let val = defaultValue;
		let hasStaticSig = false;

		if (!hasSignal) {
			if (input?.connectionValid && input.outputItem?.type === "Number") {
				val = Number(input.outputItem.data ?? defaultValue);
			} else {
				const rawVal = unwrapFrameParam(
					op[configKey],
					compositionFrame,
					fps,
				).value;
				const numVal = Number(rawVal ?? defaultValue);
				val = isNaN(numVal) ? defaultValue : numVal;
			}
		} else if (sd) {
			val = Number(sd.offset ?? 0.0);
			hasStaticSig = true;
		}

		return { val, hasStaticSig, sd };
	};

	const offsetXRes = resolveBindable("offsetX", 0);
	const offsetYRes = resolveBindable("offsetY", 0);

	// Determine edge mode numeric index
	let edgeModeIndex = 0.0; // wrap
	const edgeMode = op.edgeMode ?? (op.wrap !== false ? "wrap" : "transparent");
	if (edgeMode === "clamp") {
		edgeModeIndex = 1.0;
	} else if (edgeMode === "transparent" || op.wrap === false) {
		edgeModeIndex = 2.0;
	} else if (edgeMode === "mirror") {
		edgeModeIndex = 3.0;
	}

	// 2. Draw child media to offscreen texture
	const { texture: tmpTex } = await drawChildToTexture(args, childMedia);

	// End active pass from drawing child
	args.pass.end();

	// 3. Fetch layout and resources
	const {
		pipeline,
		uniformLayout: uLayout,
		singleTextureLayout: tLayout,
		signalTextureLayout: sigLayout,
	} = getTileOffsetResources(ctx.device, ctx.renderer.format);

	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);

	// Fetch dynamic signal texture views on-demand
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
			sd: TileOffsetSignalData | null;
		},
		suffix: string,
	) => signalTextureView(signalContext, res, `${nodeId}_${suffix}`);

	const offsetXView = getSignalView(offsetXRes, "offsetX_sig");
	const offsetYView = getSignalView(offsetYRes, "offsetY_sig");

	// 4. Fill uniform array
	uniformData[0] = offsetXRes.val;
	uniformData[1] = offsetYRes.val;
	uniformData[2] = offsetXRes.hasStaticSig ? 1.0 : 0.0;
	uniformData[3] = offsetYRes.hasStaticSig ? 1.0 : 0.0;
	uniformData[4] = width;
	uniformData[5] = height;
	uniformData[6] = edgeModeIndex;
	uniformData[7] = 0.0;

	const uniformBuffer = ctx.renderer.getTemporaryBuffer(uniformData);

	// 5. Output Texture
	const outTex = ctx.renderer.getTemporaryTexture(width, height, [
		tmpTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);

	// 6. Run tile offset pass
	drawFullscreenPass(encoder, outTex.createView(), pipeline, [
		ctx.device.createBindGroup({
			layout: uLayout,
			entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
		}),
		ctx.renderer.bindGroupCache.getBindGroup(
			ctx.device,
			tLayout,
			tmpTex,
			sampler,
		),
		ctx.device.createBindGroup({
			layout: sigLayout,
			entries: [
				{ binding: 0, resource: offsetXView },
				{ binding: 1, resource: offsetYView },
				{ binding: 2, resource: sampler },
			],
		}),
	]);

	// 7. Output result back to targetView
	drawTextureToTarget(args, outTex, op.opacity ?? 1.0);
};

export const WebGPURenderer = TileOffsetWebGPURenderer;
