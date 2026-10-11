import { describe, expect, it } from "vitest";
import { Composition, Layer, LayerAnimation, Light } from "./index.js";
import { computeMse, decodePngPixels, getPixel } from "./test-utils/pixels.js";

describe("Screen-Space 3D Normal Relighting Conformance", () => {
	it("1. Flat Fallback Invariant: Layer with relighting renders gracefully when no lights are present", async () => {
		const comp = new Composition({
			width: 200,
			height: 200,
			fps: 30,
			durationMs: 1000,
			backgroundColor: "#000000",
		});

		comp.add(
			Layer.box({
				id: "relit-card-unlit",
				x: 20,
				y: 20,
				width: 160,
				height: 160,
				background: "#4080ff",
			}).withRelighting({
				roughness: 0.4,
				specularStrength: 0.5,
			}),
		);

		const frameBuf = await comp.renderFrame({ frame: 0 });
		const img = await decodePngPixels(frameBuf);
		const center = getPixel(img, 100, 100);

		// Must render the blue color without collapsing into black
		expect(center.b).toBeGreaterThan(150);
		expect(center.a).toBe(255);
	});

	it("2. Point Light Relighting Invariant: Distance decay produces calibrated radial falloff", async () => {
		const comp = new Composition({
			width: 400,
			height: 400,
			fps: 30,
			durationMs: 1000,
			backgroundColor: "#000000",
		});

		// Ambient baseline + intense Point light at center (200, 200, 50)
		comp.add(Light.ambient("#ffffff", 0.1));
		comp.add(
			Light.point({
				color: "#ffffff",
				intensity: 2.0,
				x: 200,
				y: 200,
				z: 60,
				radius: 250,
				decay: 1.5,
			}),
		);

		comp.add(
			Layer.box({
				id: "relit-surface",
				x: 50,
				y: 50,
				width: 300,
				height: 300,
				background: "#808080",
			}).withRelighting({
				roughness: 0.3,
				specularStrength: 0.7,
				ambientIntensity: 0.1,
			}),
		);

		const frameBuf = await comp.renderFrame({ frame: 0 });
		const img = await decodePngPixels(frameBuf);

		const centerPixel = getPixel(img, 200, 200);
		const edgePixel = getPixel(img, 60, 200);
		const cornerPixel = getPixel(img, 60, 60);

		// Center right under the point light must be significantly brighter than the distant edge
		expect(centerPixel.r).toBeGreaterThan(edgePixel.r);
		expect(edgePixel.r).toBeGreaterThan(cornerPixel.r);
		expect(centerPixel.r).toBeGreaterThan(120);
	});

	it("3. Dynamic Light Motion & Specular Sweep: Light animation across timeline produces spatial shift (MSE > 0)", async () => {
		const comp = new Composition({
			width: 400,
			height: 200,
			fps: 30,
			durationMs: 1000,
			backgroundColor: "#000000",
		});

		comp.add(Light.ambient("#ffffff", 0.15));

		// Animate point light moving from x=80 to x=320 across 30 frames
		comp.add(
			Light.point({
				id: "moving-light",
				color: "#ffffff",
				intensity: 2.5,
				x: 80,
				y: 100,
				z: 50,
				radius: 200,
				decay: 1.0,
				// Sweep across over the first second (30 fps).
				animation: LayerAnimation.create().fromTo("x", 80, 320, {
					start: 0,
					end: 30,
				}),
			}),
		);

		comp.add(
			Layer.box({
				id: "wide-banner",
				x: 40,
				y: 40,
				width: 320,
				height: 120,
				background: "#606060",
			}).withRelighting({
				roughness: 0.25,
				specularStrength: 1.0,
			}),
		);

		const frame0Buf = await comp.renderFrame({ frame: 0 });
		const frame29Buf = await comp.renderFrame({ frame: 29 });

		const img0 = await decodePngPixels(frame0Buf);
		const img29 = await decodePngPixels(frame29Buf);

		// Frame 0: Peak should be near left (x=80)
		const leftPixelF0 = getPixel(img0, 80, 100);
		const rightPixelF0 = getPixel(img0, 320, 100);
		expect(leftPixelF0.r).toBeGreaterThan(rightPixelF0.r);

		// Frame 29: Peak should have swept to the right (x=320)
		const leftPixelF29 = getPixel(img29, 80, 100);
		const rightPixelF29 = getPixel(img29, 320, 100);
		expect(rightPixelF29.r).toBeGreaterThan(leftPixelF29.r);

		// Overall temporal difference must be non-zero
		const deltaMse = computeMse(img0.pixels, img29.pixels);
		expect(deltaMse).toBeGreaterThan(100);
	});
});
