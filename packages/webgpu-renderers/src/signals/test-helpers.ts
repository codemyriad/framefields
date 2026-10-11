import fs from "node:fs";
import { type Mock, vi } from "vitest";

/** GPUBufferUsage / GPUTextureUsage flags, which Node lacks without a device. */
export function installGPUUsageGlobals(): void {
	if (typeof globalThis.GPUBufferUsage === "undefined") {
		globalThis.GPUBufferUsage = {
			MAP_READ: 1,
			MAP_WRITE: 2,
			COPY_SRC: 4,
			COPY_DST: 8,
			INDEX: 16,
			VERTEX: 32,
			UNIFORM: 64,
			STORAGE: 128,
			INDIRECT: 256,
			QUERY_RESOLVE: 512,
		} as unknown as typeof GPUBufferUsage;
	}

	if (typeof globalThis.GPUTextureUsage === "undefined") {
		globalThis.GPUTextureUsage = {
			COPY_SRC: 1,
			COPY_DST: 2,
			TEXTURE_BINDING: 4,
			STORAGE_BINDING: 8,
			RENDER_ATTACHMENT: 16,
		} as unknown as typeof GPUTextureUsage;
	}
}

/** A mock device whose render pipeline draws into a `width`×`height` texture. */
export function createRenderMocks(
	width: number,
	height: number,
): {
	mockTexture: GPUTexture;
	mockBuffer: GPUBuffer;
	mockPipeline: GPURenderPipeline;
	mockPass: Record<"setPipeline" | "setBindGroup" | "draw" | "end", Mock>;
	mockEncoder: GPUCommandEncoder;
	mockDevice: GPUDevice;
} {
	const mockTexture = {
		createView: vi.fn().mockReturnValue({}),
		destroy: vi.fn(),
		width,
		height,
	} as unknown as GPUTexture;

	const mockBuffer = {
		destroy: vi.fn(),
	} as unknown as GPUBuffer;

	const mockPipeline = {
		getBindGroupLayout: vi.fn().mockReturnValue({}),
	} as unknown as GPURenderPipeline;

	const mockPass = {
		setPipeline: vi.fn(),
		setBindGroup: vi.fn(),
		draw: vi.fn(),
		end: vi.fn(),
	};

	const mockEncoder = {
		beginRenderPass: vi.fn().mockReturnValue(mockPass),
		copyBufferToTexture: vi.fn(),
		finish: vi.fn().mockReturnValue({}),
	} as unknown as GPUCommandEncoder;

	const mockDevice = {
		createTexture: vi.fn().mockReturnValue(mockTexture),
		createBuffer: vi.fn().mockReturnValue(mockBuffer),
		createSampler: vi.fn().mockReturnValue({}),
		createShaderModule: vi.fn().mockReturnValue({}),
		createRenderPipeline: vi.fn().mockReturnValue(mockPipeline),
		createBindGroup: vi.fn().mockReturnValue({}),
		createCommandEncoder: vi.fn().mockReturnValue(mockEncoder),
		queue: {
			writeBuffer: vi.fn(),
			writeTexture: vi.fn(),
			submit: vi.fn(),
		},
	} as unknown as GPUDevice;

	return {
		mockTexture,
		mockBuffer,
		mockPipeline,
		mockPass,
		mockEncoder,
		mockDevice,
	};
}

/** The channels of a 16-bit stereo PCM WAV file. */
export function loadWavChannels(filePath: string): {
	channels: Float32Array[];
	sampleRate: number;
} {
	const buf = fs.readFileSync(filePath);
	const numChannels = buf.readUInt16LE(22);
	const sampleRate = buf.readUInt32LE(24);
	let offset = 12;
	while (offset < buf.length - 8) {
		const id = buf.toString("ascii", offset, offset + 4);
		const size = buf.readUInt32LE(offset + 4);
		if (id === "data") {
			offset += 8;
			break;
		}
		offset += 8 + size;
	}
	const numSamples = Math.floor((buf.length - offset) / (numChannels * 2));
	const left = new Float32Array(numSamples);
	const right = new Float32Array(numSamples);
	for (let i = 0; i < numSamples; i++) {
		left[i] = buf.readInt16LE(offset + i * 4) / 32768.0;
		right[i] = buf.readInt16LE(offset + i * 4 + 2) / 32768.0;
	}
	return { channels: [left, right], sampleRate };
}
