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
import { solveCornerPinHomography } from "../shared/utils.js";

const cornerPinWgsl = `
struct CornerPinUniforms {
	matrix_row0 : vec4<f32>,
	matrix_row1 : vec4<f32>,
	matrix_row2 : vec4<f32>,
};

@group(0) @binding(0) var<uniform> u : CornerPinUniforms;
@group(1) @binding(0) var tex        : texture_2d<f32>;
@group(1) @binding(1) var samp       : sampler;
${FULLSCREEN_QUAD_WGSL}

@fragment fn fs(in : VSOut) -> @location(0) vec4<f32> {
	let x = in.uv.x;
	let y = in.uv.y;

	let d = u.matrix_row2.x * x + u.matrix_row2.y * y + u.matrix_row2.z;
	if (abs(d) < 1e-6) {
		return vec4<f32>(0.0);
	}

	let src_u = (u.matrix_row0.x * x + u.matrix_row0.y * y + u.matrix_row0.z) / d;
	let src_v = (u.matrix_row1.x * x + u.matrix_row1.y * y + u.matrix_row1.z) / d;

	if (src_u < 0.0 || src_u > 1.0 || src_v < 0.0 || src_v > 1.0) {
		return vec4<f32>(0.0);
	}

	let color = textureSampleLevel(tex, samp, vec2<f32>(src_u, src_v), 0.0);
	return color;
}
`;

interface DeviceCornerPinResources {
	pinPipeline: GPURenderPipeline;
	pinUniformLayout: GPUBindGroupLayout;
	pinTextureLayout: GPUBindGroupLayout;
}

const deviceResourceCache = new WeakMap<GPUDevice, DeviceCornerPinResources>();

function getCornerPinResources(device: GPUDevice, format: GPUTextureFormat) {
	let res = deviceResourceCache.get(device);
	if (!res) {
		const pinUniformLayout = fragmentBindGroupLayout(device, ["uniform"]);

		const pinTextureLayout = fragmentBindGroupLayout(device, [
			"texture",
			"sampler",
		]);

		const module = device.createShaderModule({
			label: "corner_pin.wgsl",
			code: cornerPinWgsl,
		});

		const pinPipeline = createFullscreenPipeline(device, {
			label: "CornerPinPipeline",
			module,
			layouts: [pinUniformLayout, pinTextureLayout],
			format,
			blend: STRAIGHT_ALPHA_BLEND,
		});

		res = { pinPipeline, pinUniformLayout, pinTextureLayout };
		deviceResourceCache.set(device, res);
	}
	return res;
}

export const CornerPinWebGPURenderer: WebGPUNodeRenderer = async (args) => {
	const { ctx, props, targetTexture, targetWidth, targetHeight, encoder } =
		args;
	const { virtualMedia } = props;
	const op = virtualMedia?.operation as any;
	if (!op || op.op !== "CornerPin") return;

	const childMedia = virtualMedia.children?.[0];
	if (!childMedia) return;

	const sourceWidth =
		op.originalWidth ??
		childMedia.metadata?.width ??
		props.containerWidth ??
		targetWidth;
	const sourceHeight =
		op.originalHeight ??
		childMedia.metadata?.height ??
		props.containerHeight ??
		targetHeight;

	// End incoming pass to avoid conflicts
	args.pass.end();

	// 2. Render child into a temporary offscreen texture at its native size
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

	// 3. Solve homography matrix mapping target corners to source corners
	const points = op.points || [];
	const h = solveCornerPinHomography(points);

	// 4. Setup uniform buffer with std140 layout alignment
	const uniformData = new Float32Array(12);
	uniformData[0] = h[0]; // h00
	uniformData[1] = h[1]; // h01
	uniformData[2] = h[2]; // h02
	uniformData[3] = 0; // padding

	uniformData[4] = h[3]; // h10
	uniformData[5] = h[4]; // h11
	uniformData[6] = h[5]; // h12
	uniformData[7] = 0; // padding

	uniformData[8] = h[6]; // h20
	uniformData[9] = h[7]; // h21
	uniformData[10] = h[8]; // h22 (1.0)
	uniformData[11] = 0; // padding

	const {
		pinPipeline: pipeline,
		pinUniformLayout: uLayout,
		pinTextureLayout: tLayout,
	} = getCornerPinResources(ctx.device, ctx.renderer.format);

	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);
	const uniformBuffer = ctx.renderer.getTemporaryBuffer(uniformData);

	const uniformBindGroup = ctx.device.createBindGroup({
		layout: uLayout!,
		entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
	});

	const textureBindGroup = ctx.renderer.bindGroupCache.getBindGroup(
		ctx.device,
		tLayout!,
		childTex,
		sampler,
	);

	// 5. Render warped result into a temporary offscreen texture at target size
	const outTex = ctx.renderer.getTemporaryTexture(targetWidth, targetHeight, [
		childTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);

	drawFullscreenPass(encoder, outTex.createView(), pipeline!, [
		uniformBindGroup,
		textureBindGroup,
	]);

	// 6. Draw warped texture on top of targetView (with loadOp: "load" to keep background/prev layers)
	drawTextureToTarget(args, outTex, op.opacity ?? 1);
};
