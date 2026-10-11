import { Canvas, loadImage } from "skia-canvas";
import { describe, expect, it } from "vitest";
import { Composition, Gradient, Layer } from "./index.js";

async function decode(png: Buffer) {
	const img = await loadImage(png);
	const canvas = new Canvas(img.width, img.height);
	const ctx = canvas.getContext("2d");
	ctx.drawImage(img, 0, 0);
	const { data } = ctx.getImageData(0, 0, img.width, img.height);
	return (x: number, y: number) => {
		const i = (Math.round(y) * img.width + Math.round(x)) * 4;
		return { r: data[i]!, g: data[i + 1]!, b: data[i + 2]!, a: data[i + 3]! };
	};
}

function comp() {
	return new Composition({
		width: 400,
		height: 200,
		fps: 30,
		durationMs: 1000,
		backgroundColor: "#000000",
	});
}

describe("CSS gradients as backgrounds and fills", () => {
	it("paints a multi-stop linear gradient box background", async () => {
		const c = comp();
		c.add(
			Layer.box({
				id: "bg",
				position: "absolute",
				x: 0,
				y: 0,
				width: 400,
				height: 200,
				background: "linear-gradient(to right, #ff0000, #00ff00 50%, #0000ff)",
			}),
		);
		const at = await decode(await c.renderFrame({ frame: 0 }));
		const left = at(4, 100);
		const mid = at(200, 100);
		const right = at(396, 100);
		expect(left.r).toBeGreaterThan(230);
		expect(mid.g).toBeGreaterThan(230);
		expect(mid.r).toBeLessThan(40);
		expect(right.b).toBeGreaterThan(230);
		// Perceptual blend: halfway red→green stays bright, not a muddy dark olive.
		const q = at(100, 100);
		expect(q.r + q.g).toBeGreaterThan(300);
	});

	it("keeps rounded corners on gradient backgrounds", async () => {
		const c = comp();
		c.add(
			Layer.box({
				id: "card",
				position: "absolute",
				x: 100,
				y: 20,
				width: 200,
				height: 160,
				borderRadius: 60,
				background: Gradient.linear(["#ffffff", "#ffffff"]),
			}),
		);
		const at = await decode(await c.renderFrame({ frame: 0 }));
		expect(at(102, 22).r).toBeLessThan(20); // outside the corner
		expect(at(200, 100).r).toBeGreaterThan(240);
	});

	it("fills a shape with a radial gradient fading to transparent", async () => {
		const c = comp();
		c.add(
			Layer.shape("rect", {
				id: "glow",
				position: "absolute",
				x: 100,
				y: 0,
				width: 200,
				height: 200,
				fill: Gradient.radial(["#ffffff", "transparent"], { circle: true }),
			}),
		);
		const at = await decode(await c.renderFrame({ frame: 0 }));
		expect(at(200, 100).r).toBeGreaterThan(220);
		expect(at(103, 3).r).toBeLessThan(30);
		expect(at(150, 100).r).toBeLessThan(at(200, 100).r);
	});

	it("sweeps a conic gradient clockwise from the top", async () => {
		const c = comp();
		c.add(
			Layer.shape("rect", {
				id: "wheel",
				position: "absolute",
				x: 100,
				y: 0,
				width: 200,
				height: 200,
				fill: "conic-gradient(#ff0000 0 25%, #00ff00 0 50%, #0000ff 0 75%, #ffffff 0)",
			}),
		);
		const at = await decode(await c.renderFrame({ frame: 0 }));
		const tr = at(260, 40); // top-right quadrant: first quarter turn
		const br = at(260, 160);
		const bl = at(140, 160);
		const tl = at(140, 40);
		expect(tr.r).toBeGreaterThan(200);
		expect(tr.g).toBeLessThan(60);
		expect(br.g).toBeGreaterThan(200);
		expect(bl.b).toBeGreaterThan(200);
		expect(tl.r + tl.g + tl.b).toBeGreaterThan(700);
	});
});
