/// <reference types="webgpu" />
import type { WebGPUNodeRenderer } from "@framefields/node-sdk";
import {
	createFullscreenPipeline,
	drawChildToTexture,
	drawFullscreenPass,
	drawTextureToTarget,
	FULLSCREEN_QUAD_WGSL,
	fragmentBindGroupLayout,
	STRAIGHT_ALPHA_BLEND,
} from "@framefields/node-sdk/gpu";

interface FlipOp {
	op: "Flip";
	horizontal?: boolean;
	vertical?: boolean;
	diagonal?: boolean;
	mode?:
		| "horizontal"
		| "vertical"
		| "both"
		| "diagonal"
		| "antiDiagonal"
		| "custom";
	symmetry?:
		| "none"
		| "leftToRight"
		| "rightToLeft"
		| "topToBottom"
		| "bottomToTop"
		| "quadrant";
	opacity?: number;
}

const WGSL_FLIP_SHADER = `
struct FlipUniforms {
	flipH     : f32,
	flipV     : f32,
	diagonal  : f32,
	symmetry  : f32,
};

@group(0) @binding(0) var<uniform> u : FlipUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;
${FULLSCREEN_QUAD_WGSL}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	var uv = in.uv;

	// 1. Split-Mirror / Symmetry
	if (u.symmetry > 0.5 && u.symmetry < 1.5) {
		// leftToRight: mirror left half to right half
		uv.x = select(uv.x, 1.0 - uv.x, uv.x > 0.5);
	} else if (u.symmetry > 1.5 && u.symmetry < 2.5) {
		// rightToLeft: mirror right half to left half
		uv.x = select(1.0 - uv.x, uv.x, uv.x > 0.5);
	} else if (u.symmetry > 2.5 && u.symmetry < 3.5) {
		// topToBottom: mirror top half to bottom half
		uv.y = select(uv.y, 1.0 - uv.y, uv.y > 0.5);
	} else if (u.symmetry > 3.5 && u.symmetry < 4.5) {
		// bottomToTop: mirror bottom half to top half
		uv.y = select(1.0 - uv.y, uv.y, uv.y > 0.5);
	} else if (u.symmetry > 4.5) {
		// quadrant: 4-way kaleidoscopic symmetry
		uv.x = select(uv.x, 1.0 - uv.x, uv.x > 0.5);
		uv.y = select(uv.y, 1.0 - uv.y, uv.y > 0.5);
	}

	// 2. Diagonal Swap (Transposition: x <-> y)
	if (u.diagonal > 0.5) {
		let temp = uv.x;
		uv.x = uv.y;
		uv.y = temp;
	}

	// 3. Horizontal / Vertical Flip
	if (u.flipH > 0.5) {
		uv.x = 1.0 - uv.x;
	}
	if (u.flipV > 0.5) {
		uv.y = 1.0 - uv.y;
	}

	return textureSampleLevel(tex, samp, uv, 0.0);
}
`;

interface DeviceFlipResources {
	uniformLayout: GPUBindGroupLayout;
	textureLayout: GPUBindGroupLayout;
	pipeline: GPURenderPipeline;
}

const deviceResourceCache = new WeakMap<GPUDevice, DeviceFlipResources>();
const uniformData = new Float32Array(4);

function getFlipResources(
	device: GPUDevice,
	targetFormat: GPUTextureFormat,
): DeviceFlipResources {
	const cached = deviceResourceCache.get(device);
	if (cached) return cached;

	const uniformLayout = fragmentBindGroupLayout(device, ["uniform"]);

	const textureLayout = fragmentBindGroupLayout(device, ["texture", "sampler"]);

	const shaderModule = device.createShaderModule({
		code: WGSL_FLIP_SHADER,
	});

	const pipelineLayout = device.createPipelineLayout({
		bindGroupLayouts: [uniformLayout, textureLayout],
	});

	const pipeline = createFullscreenPipeline(device, {
		module: shaderModule,
		layouts: pipelineLayout,
		format: targetFormat,
		blend: STRAIGHT_ALPHA_BLEND,
	});

	const res: DeviceFlipResources = {
		uniformLayout,
		textureLayout,
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

	const op = props.virtualMedia?.operation as FlipOp | undefined;
	if (op?.op !== "Flip" || !op) return;

	const childMedia = props.virtualMedia.children?.[0];
	if (!childMedia) return;

	pass.end();

	const width = targetWidth;
	const height = targetHeight;

	const sourceWidth =
		childMedia.metadata?.width ?? props.containerWidth ?? width;
	const sourceHeight =
		childMedia.metadata?.height ?? props.containerHeight ?? height;

	// 1. Render child into offscreen texture
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

	// 2. Fetch resources & configure flip parameters
	const {
		pipeline,
		uniformLayout: uLayout,
		textureLayout: tLayout,
	} = getFlipResources(ctx.device, ctx.renderer.format);

	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);

	const mode = op.mode ?? "horizontal";
	let isFlipH = false;
	let isFlipV = false;
	let isDiag = false;

	if (mode === "horizontal") {
		isFlipH = true;
	} else if (mode === "vertical") {
		isFlipV = true;
	} else if (mode === "both") {
		isFlipH = true;
		isFlipV = true;
	} else if (mode === "diagonal") {
		isDiag = true;
	} else if (mode === "antiDiagonal") {
		isDiag = true;
		isFlipH = true;
		isFlipV = true;
	} else if (mode === "custom") {
		isFlipH = op.horizontal ?? true;
		isFlipV = op.vertical ?? false;
		isDiag = op.diagonal ?? false;
	}

	let symVal = 0.0;
	if (op.symmetry === "leftToRight") symVal = 1.0;
	else if (op.symmetry === "rightToLeft") symVal = 2.0;
	else if (op.symmetry === "topToBottom") symVal = 3.0;
	else if (op.symmetry === "bottomToTop") symVal = 4.0;
	else if (op.symmetry === "quadrant") symVal = 5.0;

	uniformData[0] = isFlipH ? 1.0 : 0.0;
	uniformData[1] = isFlipV ? 1.0 : 0.0;
	uniformData[2] = isDiag ? 1.0 : 0.0;
	uniformData[3] = symVal;

	const uniformBuffer = ctx.renderer.getTemporaryBuffer(uniformData);

	const uBindGroup = ctx.device.createBindGroup({
		layout: uLayout,
		entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
	});

	const tBindGroup = ctx.device.createBindGroup({
		layout: tLayout,
		entries: [
			{ binding: 0, resource: childTex.createView() },
			{ binding: 1, resource: sampler },
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
