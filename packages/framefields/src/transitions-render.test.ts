import { Canvas, loadImage } from "skia-canvas";
import { describe, expect, it } from "vitest";
import {
	Composition,
	Layer,
	Transition,
	type TransitionSpec,
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

const W = 400;
const H = 200;
const scene = (id: string, background: string) =>
	Layer.box({ id, width: W, height: H, background });

/** Red scene (30 frames) then blue scene (30 frames) through `t`. */
async function render(t: TransitionSpec, frame: number) {
	const seq = Layer.sequence(
		[
			{ scene: scene("a", "#ff0000"), duration: 30 },
			{ scene: scene("b", "#0000ff"), duration: 30, transition: t },
		],
		{ width: W, height: H },
	);
	const comp = new Composition({
		width: W,
		height: H,
		fps: 30,
		durationMs: (seq.totalFrames / 30) * 1000,
		backgroundColor: "#000000",
	});
	comp.add(seq);
	return { at: await decode(await comp.renderFrame({ frame })), seq };
}

describe("Layer.sequence + Transition", () => {
	it("overlaps scenes by the transition length", async () => {
		const { seq } = await render(Transition.crossfade(10), 0);
		expect(seq.sceneStarts).toEqual([0, 20]);
		expect(seq.totalFrames).toBe(50);
		const cut = await render(Transition.cut(), 0);
		expect(cut.seq.sceneStarts).toEqual([0, 30]);
	});

	it("crossfades", async () => {
		const before = (await render(Transition.crossfade(10), 15)).at(200, 100);
		const mid = (await render(Transition.crossfade(10), 25)).at(200, 100);
		const after = (await render(Transition.crossfade(10), 35)).at(200, 100);
		expect(before.r).toBeGreaterThan(240);
		expect(mid.r).toBeGreaterThan(60);
		expect(mid.b).toBeGreaterThan(60);
		expect(after.b).toBeGreaterThan(240);
		expect(after.r).toBeLessThan(10);
	});

	it("pushes the old scene out as the new one comes in", async () => {
		const { at } = await render(Transition.push("left", 10, "none"), 25);
		// Halfway: red has moved left by half, blue fills the right half.
		expect(at(100, 100).r).toBeGreaterThan(240);
		expect(at(300, 100).b).toBeGreaterThan(240);
	});

	it("wipes with a matte edge", async () => {
		const { at } = await render(Transition.wipe("right", 10, "none"), 25);
		// Edge travelling right: the left half is already revealed.
		expect(at(100, 100).b).toBeGreaterThan(240);
		expect(at(300, 100).r).toBeGreaterThan(240);
	});

	it("opens an iris from the center", async () => {
		const { at } = await render(Transition.iris(10, { ease: "none" }), 25);
		expect(at(200, 100).b).toBeGreaterThan(240);
		expect(at(5, 5).r).toBeGreaterThan(240);
	});

	it("fades through the background", async () => {
		const { at } = await render(Transition.fadeThrough(10, "none"), 25);
		const c = at(200, 100);
		expect(c.r).toBeLessThan(40);
		expect(c.b).toBeLessThan(40);
	});
});
