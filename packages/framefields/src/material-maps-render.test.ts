import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Canvas, loadImage } from "skia-canvas";
import { afterAll, describe, expect, it } from "vitest";
import { Composition, Layer } from "./index.js";

async function decode(png: Buffer) {
	const img = await loadImage(png);
	const canvas = new Canvas(img.width, img.height);
	const ctx = canvas.getContext("2d");
	ctx.drawImage(img, 0, 0);
	const { data } = ctx.getImageData(0, 0, img.width, img.height);
	return (x: number, y: number) => {
		const i = (Math.round(y) * img.width + Math.round(x)) * 4;
		return 0.2126 * data[i]! + 0.7152 * data[i + 1]! + 0.0722 * data[i + 2]!;
	};
}

async function solidPng(rgb: [number, number, number]): Promise<string> {
	const c = new Canvas(4, 4);
	const ctx = c.getContext("2d");
	ctx.fillStyle = `rgb(${rgb.join(",")})`;
	ctx.fillRect(0, 0, 4, 4);
	return `data:image/png;base64,${(await c.toBuffer("png")).toString("base64")}`;
}

const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ff-maps-"));
afterAll(() => fs.rm(dir, { recursive: true, force: true }));

/**
 * A 400×400 quad in the z = 0 plane facing the camera (-z), u along +x and
 * v down the image (+y), with the given material maps.
 */
async function quadGltf(
	name: string,
	material: Record<string, unknown>,
	images: string[],
): Promise<string> {
	const pos = new Float32Array([
		-200, -200, 0, 200, -200, 0, -200, 200, 0, 200, 200, 0,
	]);
	const nrm = new Float32Array([0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1]);
	const uv = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
	const idx = new Uint16Array([0, 2, 1, 1, 2, 3]);
	const bin = Buffer.concat(
		[pos, nrm, uv, idx].map((a) => Buffer.from(a.buffer)),
	);
	const doc = {
		asset: { version: "2.0" },
		buffers: [
			{
				byteLength: bin.length,
				uri: `data:application/octet-stream;base64,${bin.toString("base64")}`,
			},
		],
		bufferViews: [
			{ buffer: 0, byteOffset: 0, byteLength: 48 },
			{ buffer: 0, byteOffset: 48, byteLength: 48 },
			{ buffer: 0, byteOffset: 96, byteLength: 32 },
			{ buffer: 0, byteOffset: 128, byteLength: 12 },
		],
		accessors: [
			{
				bufferView: 0,
				componentType: 5126,
				count: 4,
				type: "VEC3",
				min: [-200, -200, 0],
				max: [200, 200, 0],
			},
			{ bufferView: 1, componentType: 5126, count: 4, type: "VEC3" },
			{ bufferView: 2, componentType: 5126, count: 4, type: "VEC2" },
			{ bufferView: 3, componentType: 5123, count: 6, type: "SCALAR" },
		],
		images: images.map((uri) => ({ uri })),
		textures: images.map((_, i) => ({ source: i })),
		materials: [material],
		meshes: [
			{
				primitives: [
					{
						attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 },
						indices: 3,
						material: 0,
					},
				],
			},
		],
		nodes: [{ mesh: 0 }],
		scenes: [{ nodes: [0] }],
		scene: 0,
	};
	const file = path.join(dir, `${name}.gltf`);
	await fs.writeFile(file, JSON.stringify(doc));
	return file;
}

async function brightness(src: string, light: [number, number, number]) {
	const comp = new Composition({
		width: 400,
		height: 400,
		fps: 30,
		durationMs: 1000,
		backgroundColor: "#000",
	});
	comp.add([
		Layer.camera({
			x: 200,
			y: 200,
			z: -800,
			targetX: 200,
			targetY: 200,
			targetZ: 0,
		}),
		Layer.directionalLight({
			x: 200 + light[0],
			y: 200 + light[1],
			z: light[2],
			targetX: 200,
			targetY: 200,
			targetZ: 0,
			intensity: 1,
		}),
		Layer.gltf(src, {
			id: "quad",
			x: 200,
			y: 200,
			z: 0,
			is3D: true,
			specularIntensity: 0,
		}),
	]);
	const at = await decode(await comp.renderFrame({ frame: 0 }));
	return at(200, 200);
}

describe("glTF material maps", () => {
	it("tilts the surface along +u with the normal map's red channel", async () => {
		const src = await quadGltf("red", { normalTexture: { index: 0 } }, [
			await solidPng([224, 128, 200]),
		]);
		const fromRight = await brightness(src, [700, 0, -700]);
		const fromLeft = await brightness(src, [-700, 0, -700]);
		expect(fromRight).toBeGreaterThan(fromLeft + 25);
	});

	it("tilts toward the top of the image with the green channel (glTF +Y up)", async () => {
		const src = await quadGltf("green", { normalTexture: { index: 0 } }, [
			await solidPng([128, 224, 200]),
		]);
		const fromAbove = await brightness(src, [0, -700, -700]); // y is down
		const fromBelow = await brightness(src, [0, 700, -700]);
		expect(fromAbove).toBeGreaterThan(fromBelow + 25);
	});

	it("darkens ambient light with the occlusion map", async () => {
		const ambientOnly = async (src: string) => {
			const comp = new Composition({
				width: 400,
				height: 400,
				fps: 30,
				durationMs: 1000,
				backgroundColor: "#000",
			});
			comp.add([
				Layer.camera({
					x: 200,
					y: 200,
					z: -800,
					targetX: 200,
					targetY: 200,
					targetZ: 0,
				}),
				Layer.ambientLight("#ffffff", 1),
				Layer.gltf(src, { id: "quad", x: 200, y: 200, z: 0, is3D: true }),
			]);
			return (await decode(await comp.renderFrame({ frame: 0 })))(200, 200);
		};
		const plain = await ambientOnly(await quadGltf("plain", {}, []));
		const occluded = await ambientOnly(
			await quadGltf("ao", { occlusionTexture: { index: 0 } }, [
				await solidPng([60, 60, 60]),
			]),
		);
		expect(occluded).toBeLessThan(plain * 0.5);
	});
});
