import { describe, expect, it } from "vitest";
import { Light3D } from "../math3d/light3d.js";
import type { Mat4 } from "../math3d/matrix4.js";
import {
	type Aabb,
	fitShadowFrustum,
	transformAabb,
} from "./shadow-frustum.js";

function project(m: Mat4, p: [number, number, number]) {
	const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
	const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
	const z = m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14];
	const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
	return [x / w, y / w, z / w];
}

const box = (cx: number, cy: number, cz: number, r: number): Aabb => ({
	min: [cx - r, cy - r, cz - r],
	max: [cx + r, cy + r, cz + r],
});

describe("fitShadowFrustum", () => {
	it("fits a directional light tightly around casters, deep enough for receivers", () => {
		const caster = box(0, 0, 0, 100);
		const floor: Aabb = { min: [-5000, 200, -5000], max: [5000, 210, 5000] };
		const f = fitShadowFrustum(
			{ type: "directional", position: [0, -1000, 0], direction: [0, 1, 0] },
			[caster],
			[caster, floor],
			2048,
		);
		expect(f).not.toBeNull();
		for (const p of [caster.min, caster.max]) {
			const [x, y, z] = project(f!.viewProj, p as [number, number, number]);
			expect(Math.abs(x)).toBeLessThanOrEqual(1);
			expect(Math.abs(y)).toBeLessThanOrEqual(1);
			expect(z).toBeGreaterThanOrEqual(0);
			expect(z).toBeLessThanOrEqual(1);
		}
		// The floor below the caster is inside the depth range, and farther.
		const [, , floorZ] = project(f!.viewProj, [0, 205, 0]);
		const [, , casterZ] = project(f!.viewProj, [0, 0, 0]);
		expect(floorZ).toBeLessThanOrEqual(1);
		expect(floorZ).toBeGreaterThan(casterZ);
		// Texels go to the caster, not the 10000 px floor.
		expect(f!.texelWorldSize).toBeLessThan(0.2);
	});

	it("aims a spot light's perspective frustum along its direction", () => {
		const caster = box(0, 0, 500, 50);
		const f = fitShadowFrustum(
			{ type: "spot", position: [0, 0, 0], direction: [0, 0, 1], angle: 30 },
			[caster],
			[caster],
			1024,
		);
		const [x, y, z] = project(f!.viewProj, [0, 0, 500]);
		expect(Math.abs(x)).toBeLessThan(1e-6);
		expect(Math.abs(y)).toBeLessThan(1e-6);
		expect(z).toBeGreaterThan(0);
		expect(z).toBeLessThan(1);
	});

	it("returns null without casters", () => {
		expect(
			fitShadowFrustum(
				{ type: "directional", position: [0, 0, 0], direction: [0, 1, 0] },
				[],
				[],
				1024,
			),
		).toBeNull();
	});

	it("transforms bounds by a model matrix", () => {
		const m: Mat4 = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 10, 20, 30, 1];
		expect(transformAabb({ min: [-1, -1, -1], max: [1, 1, 1] }, m)).toEqual({
			min: [8, 18, 28],
			max: [12, 22, 32],
		});
	});
});

describe("Light3D physical block", () => {
	it("switches on physical shading and tone mapping only with an environment", () => {
		const legacy = Light3D.packLightsUniforms([
			{ type: "ambient", intensity: 1 },
		]);
		expect(legacy[6]).toBe(0);
		const physical = Light3D.packLightsUniforms([], { preset: "studio" });
		expect(physical[5]).toBe(1); // lit even with no lights
		expect(physical[6]).toBe(1);
		expect(physical[7]).toBe(3); // neutral by default
		const agx = Light3D.packLightsUniforms([], {
			toneMapping: "agx",
			exposure: 1,
		});
		expect(agx[7]).toBe(2);
		expect(agx[8 + 8 * 16 + 3]).toBeCloseTo(2); // exposure +1 stop
	});

	it("indexes the shadow light among non-ambient lights", () => {
		const lights = [
			{ type: "ambient" as const },
			{ type: "point" as const },
			{ type: "directional" as const, castShadow: true },
		];
		expect(Light3D.shadowCasterIndex(lights)).toBe(1);
		expect(
			Light3D.shadowCasterIndex([{ type: "point" as const, castShadow: true }]),
		).toBe(-1);
	});
});
