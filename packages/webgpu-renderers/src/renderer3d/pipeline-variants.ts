/**
 * The 3D pipelines draw into one of four target sets: color; color + linear
 * depth (DoF); and either of those plus an HDR radiance target (bloom). The
 * bloom variants are built on first use from the same descriptor parts.
 */

/** Premultiplied-alpha "over". */
export const PREMULTIPLIED_OVER: GPUBlendState = {
	color: {
		srcFactor: "one",
		dstFactor: "one-minus-src-alpha",
		operation: "add",
	},
	alpha: {
		srcFactor: "one",
		dstFactor: "one-minus-src-alpha",
		operation: "add",
	},
};

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
