import { Canvas, loadImage } from "skia-canvas";
import { describe, expect, it } from "vitest";
import {
	Composition,
	hoistMattes,
	Layer,
	Layer3D,
	LayerAnimation,
} from "./index.js";

async function decode(png: Buffer) {
	const img = await loadImage(png);
	const canvas = new Canvas(img.width, img.height);
	const ctx = canvas.getContext("2d");
	ctx.drawImage(img, 0, 0);
	const { data } = ctx.getImageData(0, 0, img.width, img.height);
	return (x: number, y: number) => {
		const i = (Math.round(y) * img.width + Math.round(x)) * 4;
		return { r: data[i]!, g: data[i + 1]!, b: data[i + 2]! };
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

const red = (extra: Record<string, unknown> = {}) =>
	Layer.box({
		id: "red",
		position: "absolute",
		x: 0,
		y: 0,
		width: 400,
		height: 200,
		background: "#ff0000",
		...extra,
	});

describe("track mattes", () => {
	it("hoists a matte into a sibling timed and sized like its owner", () => {
		const [owner, matte] = hoistMattes([
			{
				...red({ startFrame: 10, durationFrames: 20 }),
				matte: { node: { id: "m", kind: "box" }, mode: "luma" },
			} as never,
		]) as Array<Record<string, unknown>>;
		expect(owner.matteId).toBe("m");
		expect(owner.matte).toBeUndefined();
		expect(matte).toMatchObject({
			id: "m",
			matteOf: "red",
			matteMode: "luma",
			position: "absolute",
			x: 0,
			y: 0,
			width: 400,
			height: 200,
			startFrame: 10,
			durationFrames: 20,
		});
	});

	it("shows the owner only inside an alpha matte", async () => {
		const c = comp();
		c.add(
			red().withMatte(
				Layer.box({
					id: "window",
					position: "absolute",
					x: 100,
					y: 50,
					width: 100,
					height: 100,
					background: "#ffffff",
				}),
			),
		);
		const at = await decode(await c.renderFrame({ frame: 0 }));
		expect(at(150, 100).r).toBeGreaterThan(240); // inside the window
		expect(at(20, 20).r).toBeLessThan(10); // outside: hidden
		expect(at(150, 100).g).toBeLessThan(10); // the matte itself is not drawn
	});

	it("inverts and uses brightness", async () => {
		const inverted = comp();
		inverted.add(
			red().withMatte(
				Layer.box({
					id: "hole",
					position: "absolute",
					x: 100,
					y: 50,
					width: 100,
					height: 100,
					background: "#fff",
				}),
				"alpha-inverted",
			),
		);
		const inv = await decode(await inverted.renderFrame({ frame: 0 }));
		expect(inv(150, 100).r).toBeLessThan(10);
		expect(inv(20, 20).r).toBeGreaterThan(240);

		const luma = comp();
		luma.add(
			red().withMatte(
				Layer.box({
					id: "ramp",
					position: "absolute",
					x: 0,
					y: 0,
					width: 400,
					height: 200,
					background: "linear-gradient(to right, #000, #fff)",
				}),
				"luma",
			),
		);
		const l = await decode(await luma.renderFrame({ frame: 0 }));
		expect(l(10, 100).r).toBeLessThan(20);
		expect(l(390, 100).r).toBeGreaterThan(220);
	});

	it("animates a wipe through the matte, keyed from the owner's start", async () => {
		const c = comp();
		c.add(
			red({ startFrame: 10 }).withMatte(
				Layer.box({
					id: "wipe",
					position: "absolute",
					x: 0,
					y: 0,
					width: 400,
					height: 200,
					background: "#fff",
				}).animate(
					LayerAnimation.keys("x", [
						[0, -400],
						[20, 0],
					]),
				),
			),
		);
		const halfway = await decode(await c.renderFrame({ frame: 20 }));
		expect(halfway(100, 100).r).toBeGreaterThan(240);
		expect(halfway(300, 100).r).toBeLessThan(10);
	});

	describe("on 3D layers", () => {
		const card = (
			id: string,
			background: string,
			z: number,
			[x, y, width, height]: number[],
		) =>
			Layer.box({ id, position: "absolute", x, y, width, height, z, background });
		const scene = () => {
			const c = comp();
			c.add(
				Layer.camera({
					x: 200,
					y: 100,
					z: -1000,
					targetX: 200,
					targetY: 100,
					targetZ: 0,
				}),
			);
			return c;
		};

		it("mattes in screen space and keeps depth against the scene", async () => {
			const c = scene();
			c.add([
				card("back", "#00ff00", 300, [-400, -200, 1200, 600]),
				card("front", "#0000ff", -100, [40, 40, 120, 120]),
				card("red", "#ff0000", 50, [-100, -50, 600, 300]).withMatte(
					Layer.box({
						id: "left-half",
						position: "absolute",
						x: 0,
						y: 0,
						width: 200,
						height: 200,
						background: "#fff",
					}),
				),
			]);
			const at = await decode(await c.renderFrame({ frame: 0 }));
			expect(at(150, 120).r).toBeGreaterThan(200); // shown inside the matte
			expect(at(250, 100).g).toBeGreaterThan(200); // matted out: no depth written
			expect(at(250, 100).r).toBeLessThan(20);
			expect(at(175, 100).b).toBeGreaterThan(200); // a nearer layer still covers it
		});

		it("mattes meshes, by brightness", async () => {
			const c = scene();
			c.add(
				Layer3D.roundedBox({
					id: "cube",
					width: 300,
					height: 300,
					depth: 300,
					radius: 10,
					x: 200,
					y: 100,
					color: "#ff0000",
					material: "unlit",
				}).withMatte(
					Layer.box({
						id: "split",
						position: "absolute",
						x: 0,
						y: 0,
						width: 400,
						height: 200,
						background: "linear-gradient(to right, #000 50%, #fff 50%)",
					}),
					"luma-inverted",
				),
			);
			const at = await decode(await c.renderFrame({ frame: 0 }));
			expect(at(180, 100).r).toBeGreaterThan(200);
			expect(at(220, 100).r).toBeLessThan(20);
		});
	});
});
