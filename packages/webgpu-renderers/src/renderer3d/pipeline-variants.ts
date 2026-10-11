/**
 * The 3D pipelines draw into one of four target sets: color; color + linear
 * depth (DoF); and either of those plus an HDR radiance target (bloom). The
 * bloom variants are built on first use from the same descriptor parts.
 */

import { PREMULTIPLIED_OVER } from "../renderer2d/pipeline-utils.js";

export { PREMULTIPLIED_OVER };

/** Linear-depth MRT target (DoF) and HDR radiance target (bloom). */
export const LINEAR_DEPTH_FORMAT: GPUTextureFormat = "rgba16float";
export const HDR_FORMAT: GPUTextureFormat = "rgba16float";

export interface PipelineVariantBase {
	label: string;
	layout: GPUPipelineLayout;
	vertex: GPUVertexState;
	module: GPUShaderModule;
	format: GPUTextureFormat;
	depthStencil: GPUDepthStencilState;
	primitive: GPUPrimitiveState;
	multisample: GPUMultisampleState;
}

/** The color pipeline and the color + linear depth (DoF) pipeline of `base`. */
export function createBasePipelines(
	device: GPUDevice,
	b: PipelineVariantBase,
): { single: GPURenderPipeline; mrt: GPURenderPipeline } {
	const create = (useMrt: boolean) =>
		device.createRenderPipeline({
			label: `${b.label}${useMrt ? "Mrt" : "Single"}`,
			layout: b.layout,
			vertex: b.vertex,
			fragment: {
				module: b.module,
				entryPoint: useMrt ? "fs_mrt" : "fs_single",
				targets: [
					{ format: b.format, blend: PREMULTIPLIED_OVER },
					...(useMrt ? [{ format: LINEAR_DEPTH_FORMAT }] : []),
				],
			},
			depthStencil: b.depthStencil,
			primitive: b.primitive,
			multisample: b.multisample,
		});
	return { single: create(false), mrt: create(true) };
}

export class BloomPipelineVariants {
	private cache = new Map<boolean, GPURenderPipeline>();

	constructor(
		private device: GPUDevice,
		private base: PipelineVariantBase,
	) {}

	/** The pipeline writing color (+ linear depth when `useMrt`) + HDR radiance. */
	get(useMrt: boolean): GPURenderPipeline {
		let pipe = this.cache.get(useMrt);
		if (!pipe) {
			const b = this.base;
			pipe = this.device.createRenderPipeline({
				label: `${b.label}${useMrt ? "MrtBloom" : "Bloom"}`,
				layout: b.layout,
				vertex: b.vertex,
				fragment: {
					module: b.module,
					entryPoint: useMrt ? "fs_mrt_bloom" : "fs_single_bloom",
					targets: [
						{ format: b.format, blend: PREMULTIPLIED_OVER },
						...(useMrt ? [{ format: LINEAR_DEPTH_FORMAT }] : []),
						{ format: HDR_FORMAT, blend: PREMULTIPLIED_OVER },
					],
				},
				depthStencil: b.depthStencil,
				primitive: b.primitive,
				multisample: b.multisample,
			});
			this.cache.set(useMrt, pipe);
		}
		return pipe;
	}
}

/**
 * Writes the shared head of a model uniform block: model matrix [0..15],
 * normal matrix [16..31] (the model matrix when absent), color tint [32..35].
 */
export function writeModelTransform(
	out: Float32Array,
	modelMatrix: ArrayLike<number>,
	normalMatrix: ArrayLike<number> | undefined,
	colorTint?: ArrayLike<number> | null,
): void {
	const norm = normalMatrix ?? modelMatrix;
	const tint = colorTint ?? [1, 1, 1, 1];
	for (let i = 0; i < 16; i++) {
		out[i] = modelMatrix[i] ?? 0;
		out[16 + i] = norm[i] ?? 0;
	}
	out[32] = tint[0];
	out[33] = tint[1];
	out[34] = tint[2];
	out[35] = tint[3];
}
