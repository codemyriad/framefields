import { describe, expect, it } from "vitest";
import { Color, mixColors, parseCssColor, sampleColorStops } from "./color.js";

const round = (c: number[] | null) => c?.map((v) => Math.round(v * 255));

describe("parseCssColor", () => {
	it("parses hex, rgb, hsl, oklch and named colors", () => {
		expect(round(parseCssColor("#f80"))).toEqual([255, 136, 0, 255]);
		expect(round(parseCssColor("#ff880080"))).toEqual([255, 136, 0, 128]);
		expect(round(parseCssColor("rgb(10, 20, 30)"))).toEqual([10, 20, 30, 255]);
		expect(round(parseCssColor("rgb(10 20 30 / 50%)"))).toEqual([
			10, 20, 30, 128,
		]);
		expect(round(parseCssColor("rgba(255,0,0,0.5)"))).toEqual([255, 0, 0, 128]);
		expect(round(parseCssColor("hsl(120, 100%, 50%)"))).toEqual([
			0, 255, 0, 255,
		]);
		expect(round(parseCssColor("hsl(240deg 100% 50%)"))).toEqual([
			0, 0, 255, 255,
		]);
		expect(round(parseCssColor("oklch(1 0 0)"))).toEqual([255, 255, 255, 255]);
		expect(round(parseCssColor("oklch(0% 0 0)"))).toEqual([0, 0, 0, 255]);
		expect(round(parseCssColor("RebeccaPurple"))).toEqual([102, 51, 153, 255]);
		expect(parseCssColor("not-a-color")).toBeNull();
		expect(parseCssColor("#ggg")).toBeNull();
	});

	it("round-trips sRGB through OKLab", () => {
		for (const hex of ["#ff5a1f", "#1f6bff", "#123456", "#ffffff", "#000000"]) {
			expect(mixColors(hex, hex, 0.5)).toBe(hex);
		}
	});
});

describe("mixColors", () => {
	it("hits the endpoints and blends perceptually", () => {
		expect(mixColors("#ff0000", "#0000ff", 0)).toBe("#ff0000");
		expect(mixColors("#ff0000", "#0000ff", 1)).toBe("#0000ff");
		// OKLab midpoint of black/white is L = 0.5, darker than the sRGB average.
		expect(mixColors("#000000", "#ffffff", 0.5)).toBe("#636363");
	});

	it("fades to transparent without drifting through grey", () => {
		const mid = parseCssColor(mixColors("#ff0000", "transparent", 0.5))!;
		expect(mid[0]).toBeCloseTo(1, 2);
		expect(mid[1]).toBeCloseTo(0, 2);
		expect(mid[3]).toBeCloseTo(0.5, 1);
	});
});

describe("Color helpers", () => {
	it("builds colors from hsl and oklch", () => {
		expect(Color.hsl(0, 1, 0.5)).toBe("#ff0000");
		expect(Color.oklch(1, 0, 0)).toBe("#ffffff");
		expect(Color.rgb(255, 0, 0, 0.5)).toBe("#ff000080");
		expect(Color.hex("rgb(255 136 0)")).toBe("#ff8800");
	});

	it("samples ramps and steps", () => {
		const ramp = Color.ramp(["#000000", "#ffffff"]);
		expect(ramp(0)).toBe("#000000");
		expect(ramp(1)).toBe("#ffffff");
		expect(Color.steps(["#ff0000", "#00ff00", "#0000ff"], 3)).toEqual([
			"#ff0000",
			"#00ff00",
			"#0000ff",
		]);
		expect(
			sampleColorStops(
				[
					{ offset: 1, color: "#0000ff" },
					{ offset: 0, color: "#ff0000" },
				],
				0,
			),
		).toBe("#ff0000");
	});

	it("adjusts lightness, alpha and contrast", () => {
		const base = "#3366cc";
		const lighter = parseCssColor(Color.lighten(base, 0.2))!;
		const orig = parseCssColor(base)!;
		expect(lighter[0] + lighter[1] + lighter[2]).toBeGreaterThan(
			orig[0] + orig[1] + orig[2],
		);
		expect(Color.alpha("#000", 0.4)).toBe("#00000066");
		expect(Color.contrast("#000000", "#ffffff")).toBeCloseTo(21, 0);
		expect(Color.readableOn("#111111")).toBe("#ffffff");
		expect(Color.readableOn("#f5f5f5")).toBe("#000000");
		expect(Color.saturate("#ff0000", 0)).toMatch(/^#([0-9a-f]{2})\1\1$/);
	});
});
