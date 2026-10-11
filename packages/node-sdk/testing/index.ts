/// <reference types="@webgpu/types" />
// Test-only helpers for node packages; not part of the runtime SDK.
import type { VirtualMediaData } from "@framefields/core";
import { ensureDevice } from "@framefields/webgpu-renderers";

/**
 * The shared WebGPU device, or `undefined` when this machine has none, so a
 * suite can `describe.skipIf(!device)` instead of failing. Only device
 * acquisition is guarded: errors inside the tests still fail them.
 */
export async function tryEnsureDevice(): Promise<GPUDevice | undefined> {
	try {
		return await ensureDevice();
	} catch (err) {
		console.warn(
			`Skipping WebGPU tests: no device (${(err as Error)?.message ?? err})`,
		);
		return undefined;
	}
}

/** An audio VirtualMediaData whose operation is `operation`. */
export function audioTestMedia(
	id: string,
	operation: Record<string, unknown>,
	metadata?: Record<string, unknown>,
): VirtualMediaData {
	return {
		id,
		type: "audio",
		...(metadata && { metadata }),
		operation,
	} as unknown as VirtualMediaData;
}

/** `channelCount` channels of `numSamples` samples, all set to `amplitude`. */
export function constantChannels(
	numSamples: number,
	amplitude = 1.0,
	channelCount = 1,
): Float32Array[] {
	return Array.from({ length: channelCount }, () =>
		new Float32Array(numSamples).fill(amplitude),
	);
}
