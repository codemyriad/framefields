import { vi } from "vitest";
import { installDOMGeometryGlobals } from "../headless/dom-matrix.js";

export interface MockTexture {
	width: number;
	height: number;
	format: string;
	usage: number;
	destroy: any;
	createView: any;
}

export interface MockBuffer {
	size: number;
	usage: number;
	destroy: any;
	getMappedRange: any;
	unmap: any;
}

export function createMockTexture(
	width = 1024,
	height = 1024,
	format = "rgba8unorm",
	usage = 0,
): MockTexture {
	return {
		width,
		height,
		format,
		usage,
		destroy: vi.fn(),
		createView: vi.fn().mockImplementation(() => ({})),
	};
}

export function createMockBuffer(size: number, usage: number): MockBuffer {
	let mappedBuffer: ArrayBuffer | null = null;
	return {
		size,
		usage,
		destroy: vi.fn(),
		getMappedRange: vi.fn().mockImplementation(() => {
			if (!mappedBuffer) mappedBuffer = new ArrayBuffer(size);
			return mappedBuffer;
		}),
		unmap: vi.fn(),
	};
}

export function createMockDevice(): GPUDevice {
	const onSubmittedWorkDonePromise = Promise.resolve();
	const queue = {
		writeBuffer: vi.fn(),
		writeTexture: vi.fn(),
		copyExternalImageToTexture: vi.fn(),
		onSubmittedWorkDone: vi.fn().mockReturnValue(onSubmittedWorkDonePromise),
	};

	const device = {
		queue,
		limits: {
			maxTextureDimension2D: 8192,
		},
		features: {
			has: vi.fn().mockReturnValue(true),
		},
		createBuffer: vi
			.fn()
			.mockImplementation((desc) => createMockBuffer(desc.size, desc.usage)),
		createTexture: vi.fn().mockImplementation((desc) => {
			const w = Array.isArray(desc.size)
				? desc.size[0]
				: (desc.size?.width ?? 1024);
			const h = Array.isArray(desc.size)
				? desc.size[1]
				: (desc.size?.height ?? 1024);
			return createMockTexture(w, h, desc.format, desc.usage);
		}),
		createSampler: vi.fn().mockImplementation(() => ({})),
		createShaderModule: vi.fn().mockImplementation(() => ({})),
		createBindGroupLayout: vi.fn().mockImplementation(() => ({})),
		createPipelineLayout: vi.fn().mockImplementation(() => ({})),
		createRenderPipeline: vi.fn().mockImplementation(() => ({})),
		createComputePipeline: vi.fn().mockImplementation(() => ({})),
		createBindGroup: vi.fn().mockImplementation(() => ({})),
	};

	return device as unknown as GPUDevice;
}

export function createMockComputePassEncoder(): GPUComputePassEncoder {
	return {
		setPipeline: vi.fn(),
		setBindGroup: vi.fn(),
		dispatchWorkgroups: vi.fn(),
		end: vi.fn(),
	} as unknown as GPUComputePassEncoder;
}

export function createMockRenderPassEncoder(): GPURenderPassEncoder {
	return {
		setScissorRect: vi.fn(),
		setPipeline: vi.fn(),
		setBindGroup: vi.fn(),
		setVertexBuffer: vi.fn(),
		setIndexBuffer: vi.fn(),
		draw: vi.fn(),
		drawIndexed: vi.fn(),
		end: vi.fn(),
	} as unknown as GPURenderPassEncoder;
}

export function createMockCommandEncoder(): GPUCommandEncoder {
	return {
		beginRenderPass: vi
			.fn()
			.mockImplementation(() => createMockRenderPassEncoder()),
		beginComputePass: vi
			.fn()
			.mockImplementation(() => createMockComputePassEncoder()),
		finish: vi.fn().mockImplementation(() => ({})),
	} as unknown as GPUCommandEncoder;
}

export function ensureDOMGlobals() {
	installDOMGeometryGlobals();
}

/** Minimal GPU flag constants for code that ORs usage/visibility bits. */
export function installMockGPUFlags(): void {
	globalThis.GPUBufferUsage = {
		UNIFORM: 1,
		VERTEX: 2,
		COPY_DST: 4,
	} as any;
	globalThis.GPUShaderStage = {
		VERTEX: 1,
		FRAGMENT: 2,
	} as any;
	globalThis.GPUTextureUsage = {
		TEXTURE_BINDING: 1,
		COPY_DST: 2,
		RENDER_ATTACHMENT: 4,
		COPY_SRC: 8,
	} as any;
}
