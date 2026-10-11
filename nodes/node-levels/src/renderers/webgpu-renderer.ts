import type { WebGPUNodeRenderer } from "@framefields/node-sdk";
import {
	createFullscreenPipeline,
	drawChildToTexture,
	drawFullscreenPass,
	drawTextureToTarget,
	fragmentBindGroupLayout,
} from "@framefields/node-sdk/gpu";
import { colorMatrixWgsl } from "@framefields/webgpu-renderers";
import {
	defaultLevelChannel,
	type LevelChannel,
	type LevelsOperation,
} from "../shared/config.js";

interface DeviceLevelsResources {
	colorMatrixPipeline: GPURenderPipeline;
	colorMatrixUniformLayout: GPUBindGroupLayout;
	singleTextureLayout: GPUBindGroupLayout;
}

const deviceResourceCache = new WeakMap<GPUDevice, DeviceLevelsResources>();
const colorMatrixData = new Float32Array(20);

function getColorMatrixResources(device: GPUDevice, format: GPUTextureFormat) {
	let res = deviceResourceCache.get(device);
	if (!res) {
		const colorMatrixUniformLayout = fragmentBindGroupLayout(device, [
			"uniform",
		]);

		const singleTextureLayout = fragmentBindGroupLayout(device, [
			"texture",
			"sampler",
		]);

		const colorMatrixModule = device.createShaderModule({
			label: "color-matrix.wgsl",
			code: colorMatrixWgsl,
		});

		const colorMatrixPipeline = createFullscreenPipeline(device, {
			label: "ColorMatrixPipeline",
			module: colorMatrixModule,
			layouts: [colorMatrixUniformLayout, singleTextureLayout],
			format,
		});

		res = {
			colorMatrixPipeline,
			colorMatrixUniformLayout,
			singleTextureLayout,
		};
		deviceResourceCache.set(device, res);
	}
	return res;
}

function getTransform(ch: LevelChannel) {
	const inRange = Math.max(ch.inWhite - ch.inBlack, 1e-5);
	const outRange = ch.outWhite - ch.outBlack;
	const scale = outRange / inRange;
	const bias = ch.outBlack - ch.inBlack * scale;
	return { scale, bias };
}

export const LevelsWebGPURenderer: WebGPUNodeRenderer = async (args) => {
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
		| LevelsOperation
		| undefined;

	if (op?.op !== "Levels" || !props.virtualMedia?.children?.[0]) return;

	pass.end();

	const childMedia = props.virtualMedia.children[0];

	const m = getTransform(op.master ?? defaultLevelChannel);
	const r = getTransform(op.red ?? defaultLevelChannel);
	const g = getTransform(op.green ?? defaultLevelChannel);
	const b = getTransform(op.blue ?? defaultLevelChannel);

	const rS = r.scale * m.scale;
	const rB = r.bias * m.scale + m.bias;

	const gS = g.scale * m.scale;
	const gB = g.bias * m.scale + m.bias;

	const bS = b.scale * m.scale;
	const bB = b.bias * m.scale + m.bias;

	const matrix = [
		rS,
		0,
		0,
		0,
		rB,
		0,
		gS,
		0,
		0,
		gB,
		0,
		0,
		bS,
		0,
		bB,
		0,
		0,
		0,
		1,
		0,
	];

	const width = targetWidth;
	const height = targetHeight;

	const { texture: tmpTex } = await drawChildToTexture(args, childMedia);

	// Apply Color Matrix
	const {
		colorMatrixPipeline: pipeline,
		colorMatrixUniformLayout: uLayout,
		singleTextureLayout: tLayout,
	} = getColorMatrixResources(ctx.device, ctx.renderer.format);
	const sampler = ctx.renderer.samplerCache.getSampler(ctx.device);

	// Remap matrix to uniform layout
	colorMatrixData[0] = matrix[0];
	colorMatrixData[1] = matrix[1];
	colorMatrixData[2] = matrix[2];
	colorMatrixData[3] = matrix[3];

	colorMatrixData[4] = matrix[5];
	colorMatrixData[5] = matrix[6];
	colorMatrixData[6] = matrix[7];
	colorMatrixData[7] = matrix[8];

	colorMatrixData[8] = matrix[10];
	colorMatrixData[9] = matrix[11];
	colorMatrixData[10] = matrix[12];
	colorMatrixData[11] = matrix[13];

	colorMatrixData[12] = matrix[15];
	colorMatrixData[13] = matrix[16];
	colorMatrixData[14] = matrix[17];
	colorMatrixData[15] = matrix[18];

	colorMatrixData[16] = matrix[4];
	colorMatrixData[17] = matrix[9];
	colorMatrixData[18] = matrix[14];
	colorMatrixData[19] = matrix[19];

	const buffer = ctx.renderer.getTemporaryBuffer(colorMatrixData);

	const outTex = ctx.renderer.getTemporaryTexture(width, height, [
		tmpTex,
		targetTexture,
		...(props.excludeTextures || []),
	]);
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
	]);

	drawTextureToTarget(args, outTex, props.opacity ?? 1);
};
