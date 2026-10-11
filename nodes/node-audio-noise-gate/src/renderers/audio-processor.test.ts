import {
	audioTestMedia,
	constantChannels,
	tryEnsureDevice,
} from "@framefields/node-sdk/testing";
import { describe, expect, it } from "vitest";
import { noiseGateAudioProcessor } from "./audio-processor.js";

const createVirtualMedia = (overrides = {}) =>
	audioTestMedia("test-media", {
		threshold: -40, // -40dB threshold
		attack: 0.01, // 10ms
		hold: 0.05, // 50ms
		release: 0.1, // 100ms
		range: -80, // -80dB reduction when closed
		...overrides,
	});

const maybeDevice = await tryEnsureDevice();

describe.skipIf(!maybeDevice)("Noise Gate Audio Processor", () => {
	const SAMPLE_RATE = 1000; // 1000 Hz makes math easy: 1ms = 1 sample
	const CLOSED_GAIN = 10 ** (-80 / 20); // 0.0001

	// Amplitudes
	const LOW_SIGNAL = 0.001; // -60dB (Below -40dB threshold)
	const HIGH_SIGNAL = 0.1; // -20dB (Above -40dB threshold)

	const device = maybeDevice as NonNullable<typeof maybeDevice>;

	it("should keep the gate CLOSED and attenuate signals below the threshold", async () => {
		const channels = constantChannels(100, LOW_SIGNAL);
		const virtualMedia = createVirtualMedia();

		await noiseGateAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-noise-gate-attenuate",
		});

		// Every sample should be attenuated by the CLOSED_GAIN
		expect(channels[0][0]).toBeCloseTo(LOW_SIGNAL * CLOSED_GAIN, 6);
		expect(channels[0][99]).toBeCloseTo(LOW_SIGNAL * CLOSED_GAIN, 6);
	});

	it("should transition to ATTACK and then OPEN when signal exceeds threshold", async () => {
		const channels = constantChannels(20, HIGH_SIGNAL);
		const virtualMedia = createVirtualMedia();

		await noiseGateAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-noise-gate-open",
		});

		// Check that the last sample is passed through with less attenuation or none
		expect(channels[0][19]).toBeCloseTo(HIGH_SIGNAL, 3);
	});
});
