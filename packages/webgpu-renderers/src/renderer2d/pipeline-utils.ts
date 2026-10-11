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

/** One uniform buffer at binding 0, visible to both shader stages. */
export function createUniformLayout(device: GPUDevice): GPUBindGroupLayout {
	return device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: { type: "uniform" },
			},
		],
	});
}

/**
 * A pipeline drawing float32x2 vertices (`vs`/`fs`) with one uniform bind
 * group into a single `format` target.
 */
export function createVec2Pipeline(
	device: GPUDevice,
	options: {
		label: string;
		module: GPUShaderModule;
		uniformLayout: GPUBindGroupLayout;
		format: GPUTextureFormat;
		topology: GPUPrimitiveTopology;
		blend?: GPUBlendState;
	},
): GPURenderPipeline {
	const { module } = options;
	return device.createRenderPipeline({
		label: options.label,
		layout: device.createPipelineLayout({
			bindGroupLayouts: [options.uniformLayout],
		}),
		vertex: {
			module,
			entryPoint: "vs",
			buffers: [
				{
					arrayStride: 8,
					attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
				},
			],
		},
		fragment: {
			module,
			entryPoint: "fs",
			targets: [
				{ format: options.format, blend: options.blend ?? PREMULTIPLIED_OVER },
			],
		},
		primitive: { topology: options.topology },
	});
}
