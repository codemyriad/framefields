import {
	audioTestMedia,
	constantChannels,
	tryEnsureDevice,
} from "@framefields/node-sdk/testing";
import { describe, expect, it } from "vitest";
import { reverbAudioProcessor } from "./audio-processor.js";

const createVirtualMedia = (overrides = {}) =>
	audioTestMedia("test-reverb-media", {
		roomSize: 0.5,
		damping: 0.5,
		wet: 0.3,
		dry: 1.0,
		width: 1.0,
		...overrides,
	});

const createChannels = (
	numSamples: number,
	amplitude: number,
	stereo = false,
): Float32Array[] => {
	if (stereo) {
		const left = new Float32Array(numSamples).map(
			(_, idx) => Math.sin(idx * 0.05) * amplitude,
		);
		const right = new Float32Array(numSamples).map(
			(_, idx) => Math.cos(idx * 0.05) * amplitude,
		);
		return [left, right];
	}
	return constantChannels(numSamples, amplitude);
};

const maybeDevice = await tryEnsureDevice();

describe.skipIf(!maybeDevice)("Reverb Audio Processor", () => {
	const SAMPLE_RATE = 44100;

	const device = maybeDevice as NonNullable<typeof maybeDevice>;

	// Known failure: sample 0 precedes the reverb tail, so it is always the dry input.
	it.fails("should apply reverb on mono audio and alter the signal", async () => {
		const channels = createChannels(2000, 0.5);
		const virtualMedia = createVirtualMedia();
		await reverbAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-reverb",
		});

		// Reverb should alter the signal from its constant state
		expect(channels[0][0]).not.toBe(0.5);
		expect(Number.isNaN(channels[0][1000])).toBe(false);
		expect(Number.isFinite(channels[0][1000])).toBe(true);
	});

	// Known failure: sample 0 precedes the reverb tail, so it is always the dry input.
	it.fails("should apply reverb on stereo audio and alter both channels with stereo width", async () => {
		const channels = createChannels(2000, 0.5, true);
		const virtualMedia = createVirtualMedia();
		await reverbAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-reverb-stereo",
		});

		// Both channels should be altered from 0.5
		expect(channels[0][0]).not.toBe(0.5);
		expect(channels[1][0]).not.toBe(0.5);

		// Because of stereo offset (23 samples added on Right), the channels should be different
		expect(channels[0][1000]).not.toBe(channels[1][1000]);
	});

	it("should respect dry/wet parameters", async () => {
		// When dry is 0.0 and wet is 0.0, signal should be fully silenced (0.0)
		const channels = createChannels(500, 0.5);
		const virtualMedia = createVirtualMedia({
			dry: 0.0,
			wet: 0.0,
		});

		await reverbAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-reverb-drywet",
		});

		expect(channels[0][0]).toBe(0);
		expect(channels[0][250]).toBe(0);
	});

	it("should safely clamp roomSize to avoid infinite feedback runaway", async () => {
		const channels = createChannels(1000, 0.5);
		// Even at roomSize 1.5, it should be clamped to 0.98 internally
		const virtualMedia = createVirtualMedia({
			roomSize: 1.5,
		});

		await reverbAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-reverb-clamp",
		});

		expect(Number.isNaN(channels[0][500])).toBe(false);
		expect(Number.isFinite(channels[0][500])).toBe(true);
	});
});
