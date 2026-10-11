import { Canvas, loadImage } from "skia-canvas";
import { describe, expect, it } from "vitest";
import { Composition, Layer, Layer3D, Lighting } from "./index.js";

async function decode(png: Buffer) {
	const img = await loadImage(png);
	const canvas = new Canvas(img.width, img.height);
	const ctx = canvas.getContext("2d");
	ctx.drawImage(img, 0, 0);
	const { data } = ctx.getImageData(0, 0, img.width, img.height);
	return {
		at(x: number, y: number) {
			const i = (Math.round(y) * img.width + Math.round(x)) * 4;
			return { r: data[i]!, g: data[i + 1]!, b: data[i + 2]! };
		},
	};
}

const lum = (c: { r: number; g: number; b: number }) =>
	0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;

/** Camera above-front of (400, 300, 0), a floor at y 360. */
function stage(world: unknown[] = []) {
	const comp = new Composition({
		width: 800,
		height: 600,
		fps: 30,
		durationMs: 1000,
		backgroundColor: "#101012",
	});
	comp.add([
		Layer.camera({
			x: 400,
			y: 300 - 500,
			z: -700,
			targetX: 400,
			targetY: 300,
			targetZ: 0,
		}),
		...(world as never[]),
	]);
	return comp;
}

const floor = () =>
	Layer3D.roundedBox({
		id: "floor",
		width: 3000,
		height: 10,
		depth: 3000,
		radius: 2,
		x: 400,
		y: 365,
		color: "#808080",
	});

describe("physical shading (Layer.environment)", () => {
	it("keeps a brand color's hue under the default neutral tone mapping", async () => {
		const comp = stage([
			...Lighting.studio({ target: [400, 300, 0], shadows: false }),
		]);
		comp.add(
			Layer3D.roundedBox({
				id: "box",
				width: 220,
				height: 220,
				depth: 220,
				radius: 20,
				x: 400,
				y: 300,
				color: "#ff5a1f",
			}),
		);
		const img = await decode(await comp.renderFrame({ frame: 0 }));
		const c = img.at(400, 300);
		expect(c.r).toBeGreaterThan(180);
		expect(c.r).toBeGreaterThan(c.g * 1.6);
		expect(c.g).toBeGreaterThan(c.b);
	});

	it("makes a polished metal reflect the environment instead of a flat tint", async () => {
		const render = async (metallic: number) => {
			const comp = stage([Layer.environment("studio")]);
			comp.add(
				Layer3D.sphere({
					id: "ball",
					radius: 160,
					x: 400,
					y: 300,
					color: "#d0d0d0",
					metallic,
					roughness: 0.1,
				}),
			);
			return decode(await comp.renderFrame({ frame: 0 }));
		};
		const metal = await render(1);
		const plastic = await render(0);
		// A mirror ball is dark where it reflects the dark floor and bright in
		// the softbox reflections: far more contrast than a matte ball.
		const samples = [
			[400, 200],
			[400, 300],
			[400, 420],
			[310, 300],
			[490, 300],
		] as const;
		const spread = (img: Awaited<ReturnType<typeof decode>>) => {
			const ls = samples.map(([x, y]) => lum(img.at(x, y)));
			return Math.max(...ls) - Math.min(...ls);
		};
		expect(spread(metal)).toBeGreaterThan(spread(plastic));
	});

	it("casts a shadow from a castShadow light onto the floor", async () => {
		const render = async (castShadow: boolean) => {
			const comp = stage([
				Layer.environment("neutral", { intensity: 0.3 }),
				Layer.directionalLight({
					x: 400,
					y: -800,
					z: 0,
					targetX: 400,
					targetY: 360,
					targetZ: 0,
					intensity: 1.5,
					castShadow,
				}),
				floor(),
				Layer3D.roundedBox({
					id: "occluder",
					width: 200,
					height: 20,
					depth: 200,
					radius: 4,
					x: 400,
					y: 150,
					color: "#808080",
				}),
			]);
			return decode(await comp.renderFrame({ frame: 0 }));
		};
		const lit = await render(false);
		const shadowed = await render(true);
		// The floor straight below the occluder (seen past its front edge).
		const below = (img: Awaited<ReturnType<typeof decode>>) =>
			lum(img.at(400, 340));
		expect(below(shadowed)).toBeLessThan(below(lit) * 0.75);
		// Floor well away from the occluder is untouched.
		expect(
			Math.abs(lum(shadowed.at(120, 520)) - lum(lit.at(120, 520))),
		).toBeLessThan(6);
	});

	it("lights a dark surface with emissive and blooms past it", async () => {
		const render = async (bloom: boolean) => {
			const comp = stage([Layer.environment("night", { bloom })]);
			comp.add(
				Layer3D.sphere({
					id: "lamp",
					radius: 80,
					x: 400,
					y: 300,
					color: "#111111",
					emissive: "#3388ff",
					emissiveIntensity: 8,
				}),
			);
			return decode(await comp.renderFrame({ frame: 0 }));
		};
		const plain = await render(false);
		const glowing = await render(true);
		const core = plain.at(400, 300);
		expect(core.b).toBeGreaterThan(200);
		expect(core.b).toBeGreaterThan(core.r);
		// Just outside the sphere: dark without bloom, blue with it.
		const halo = glowing.at(400 + 110, 300);
		const bare = plain.at(400 + 110, 300);
		expect(halo.b).toBeGreaterThan(bare.b + 15);
		expect(halo.b).toBeGreaterThan(halo.r);
	});

	it("fades distant surfaces into the fog color", async () => {
		const comp = stage([
			Layer.environment("soft", {
				fog: { color: "#ff0000", start: 600, end: 1500 },
			}),
			floor(),
		]);
		const img = await decode(await comp.renderFrame({ frame: 0 }));
		const far = img.at(400, 180);
		const near = img.at(400, 590);
		expect(far.r).toBeGreaterThan(far.g + 80);
		expect(Math.abs(near.r - near.g)).toBeLessThan(25);
	});
});
