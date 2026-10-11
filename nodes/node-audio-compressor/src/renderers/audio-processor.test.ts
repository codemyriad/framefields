import {
	audioTestMedia,
	constantChannels,
	tryEnsureDevice,
} from "@framefields/node-sdk/testing";
import { describe, expect, it } from "vitest";
import { compressorAudioProcessor } from "./audio-processor.js";

const createVirtualMedia = (overrides = {}) =>
	audioTestMedia("test-compressor-media", {
		threshold: -24,
		ratio: 4,
		attack: 0.003,
		release: 0.25,
		knee: 6,
		makeupGain: 0,
		...overrides,
	});

const maybeDevice = await tryEnsureDevice();

describe.skipIf(!maybeDevice)("Compressor Audio Processor", () => {
	const SAMPLE_RATE = 44100;

	const device = maybeDevice as NonNullable<typeof maybeDevice>;

	it("should process mono audio and not crash", async () => {
		const channels = constantChannels(1000, 0.5);
		const virtualMedia = createVirtualMedia();

		await compressorAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-compressor-mono",
		});

		expect(channels[0][0]).toBeLessThanOrEqual(0.5);
		expect(Number.isNaN(channels[0][0])).toBe(false);
	});

	it("should apply makeup gain correctly", async () => {
		// A small input amplitude (0.01) is below threshold -24dBFS, so no compression occurs.
		// With makeupGain = 6dB, output amplitude should be 0.01 * 10^(6/20) ~ 0.02.
		const channels = constantChannels(1000, 0.01);
		const virtualMedia = createVirtualMedia({
			threshold: -24,
			makeupGain: 6.0,
		});

		await compressorAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-compressor-makeup",
		});

		expect(channels[0][0]).toBeCloseTo(0.02, 3);
	});

	it("should compress high amplitude signals above threshold", async () => {
		// A very large amplitude (0.8) is far above threshold -24dBFS.
		// With ratio 4, it should be compressed to a lower level.
		const channels = constantChannels(1000, 0.8);
		const virtualMedia = createVirtualMedia({
			threshold: -24,
			ratio: 4.0,
			makeupGain: 0.0,
			attack: 0.001,
		});

		await compressorAudioProcessor(channels, SAMPLE_RATE, virtualMedia, {
			device,
			frame: 0,
			fps: 24,
			renderId: "test-compressor-reduction",
		});

		expect(channels[0][999]).toBeLessThan(0.8);
		expect(channels[0][999]).toBeGreaterThan(0.0);
	});
});
