/** The per-draw matte binding of the 3D pipelines (see `shaders/matte3d.ts`). */
export function matteLayoutEntry(binding: number): GPUBindGroupLayoutEntry {
	return {
		binding,
		visibility: GPUShaderStage.FRAGMENT,
		texture: { sampleType: "float" },
	};
}

const fullCoverage = new WeakMap<GPUDevice, GPUTextureView>();

/** A 1×1 opaque white view: full coverage, for draws without a matte. */
export function fullCoverageView(device: GPUDevice): GPUTextureView {
	let view = fullCoverage.get(device);
	if (!view) {
		const tex = device.createTexture({
			label: "Matte3DFullCoverage",
			size: [1, 1],
			format: "rgba8unorm",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
		});
		device.queue.writeTexture(
			{ texture: tex },
			new Uint8Array([255, 255, 255, 255]),
			{ bytesPerRow: 4 },
			[1, 1],
		);
		view = tex.createView();
		fullCoverage.set(device, view);
	}
	return view;
}
