/**
 * CSS color parsing, perceptual mixing and authoring helpers, shared by the
 * schema evaluator (color tracks), the GPU renderers (fills) and the public
 * `Color` helper. Pure and dependency-free.
 *
 * Every helper returns a CSS string any color prop accepts: `#rrggbb`, or
 * `#rrggbbaa` when not fully opaque.
 */

/** Straight (non-premultiplied) sRGB, each channel 0..1. */
export type Rgba = [r: number, g: number, b: number, a: number];

const NAMED: Record<string, string> = {
	transparent: "#00000000",
	black: "#000000",
	white: "#ffffff",
	red: "#ff0000",
	green: "#008000",
	lime: "#00ff00",
	blue: "#0000ff",
	yellow: "#ffff00",
	cyan: "#00ffff",
	aqua: "#00ffff",
	magenta: "#ff00ff",
	fuchsia: "#ff00ff",
	orange: "#ffa500",
	purple: "#800080",
	pink: "#ffc0cb",
	gray: "#808080",
	grey: "#808080",
	silver: "#c0c0c0",
	maroon: "#800000",
	navy: "#000080",
	teal: "#008080",
	olive: "#808000",
	gold: "#ffd700",
	coral: "#ff7f50",
	salmon: "#fa8072",
	crimson: "#dc143c",
	indigo: "#4b0082",
	violet: "#ee82ee",
	turquoise: "#40e0d0",
	tomato: "#ff6347",
	chocolate: "#d2691e",
	tan: "#d2b48c",
	beige: "#f5f5dc",
	ivory: "#fffff0",
	khaki: "#f0e68c",
	lavender: "#e6e6fa",
	mint: "#98ff98",
	skyblue: "#87ceeb",
	slategray: "#708090",
	dimgray: "#696969",
	lightgray: "#d3d3d3",
	darkgray: "#a9a9a9",
	whitesmoke: "#f5f5f5",
	hotpink: "#ff69b4",
	deeppink: "#ff1493",
	dodgerblue: "#1e90ff",
	royalblue: "#4169e1",
	steelblue: "#4682b4",
	midnightblue: "#191970",
	limegreen: "#32cd32",
	forestgreen: "#228b22",
	seagreen: "#2e8b57",
	darkorange: "#ff8c00",
	orangered: "#ff4500",
	firebrick: "#b22222",
	rebeccapurple: "#663399",
};

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function parseHex(h: string): Rgba | null {
	if (!/^[0-9a-f]+$/i.test(h)) return null;
	const x = (i: number, n: number) =>
		parseInt(n === 1 ? h[i] + h[i] : h.slice(i, i + 2), 16) / 255;
	switch (h.length) {
		case 3:
			return [x(0, 1), x(1, 1), x(2, 1), 1];
		case 4:
			return [x(0, 1), x(1, 1), x(2, 1), x(3, 1)];
		case 6:
			return [x(0, 2), x(2, 2), x(4, 2), 1];
		case 8:
			return [x(0, 2), x(2, 2), x(4, 2), x(6, 2)];
		default:
			return null;
	}
}

/** Splits `rgb(1 2 3 / 50%)` or `rgb(1, 2, 3, .5)` into its argument tokens. */
function fnArgs(body: string): string[] {
	return body
		.replace(/\s*\/\s*/, " ")
		.split(/[\s,]+/)
		.filter(Boolean);
}

/** A number, or a percentage scaled so 100% = `pct100`. */
function num(tok: string | undefined, pct100: number): number {
	if (tok === undefined || tok === "none") return 0;
	if (tok.endsWith("%")) return (parseFloat(tok) / 100) * pct100;
	return parseFloat(tok);
}

function hue(tok: string | undefined): number {
	if (tok === undefined) return 0;
	if (tok.endsWith("turn")) return parseFloat(tok) * 360;
	if (tok.endsWith("rad")) return (parseFloat(tok) * 180) / Math.PI;
	return parseFloat(tok);
}

function alpha(tok: string | undefined): number {
	return tok === undefined ? 1 : clamp01(num(tok, 1));
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
	const a = s * Math.min(l, 1 - l);
	const f = (n: number) => {
		const k = (n + h / 30) % 12;
		return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
	};
	return [f(0), f(8), f(4)];
}

// ── OKLab (Björn Ottosson) ─────────────────────────────────────────────

const toLinear = (c: number) =>
	c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
const toGamma = (c: number) =>
	c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;

/** sRGB (0..1) → OKLab [L, a, b]. */
export function rgbToOklab(
	r: number,
	g: number,
	b: number,
): [number, number, number] {
	const lr = toLinear(r);
	const lg = toLinear(g);
	const lb = toLinear(b);
	const l = Math.cbrt(
		0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb,
	);
	const m = Math.cbrt(
		0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb,
	);
	const s = Math.cbrt(
		0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb,
	);
	return [
		0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
		1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
		0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
	];
}

/** OKLab → sRGB (0..1, clamped to gamut). */
export function oklabToRgb(
	L: number,
	a: number,
	b: number,
): [number, number, number] {
	const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
	const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
	const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
	return [
		clamp01(toGamma(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s)),
		clamp01(toGamma(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s)),
		clamp01(toGamma(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)),
	];
}

/**
 * Parses any CSS color this engine accepts: hex (3/4/6/8), `rgb()`/`rgba()`,
 * `hsl()`/`hsla()`, `oklch()`, `oklab()` (comma or space syntax, `/ alpha`)
 * and common named colors. Returns `null` when the string is not a color.
 */
export function parseCssColor(input: string): Rgba | null {
	const s = input.trim().toLowerCase();
	if (!s) return null;
	if (s.startsWith("#")) return parseHex(s.slice(1));
	const named = NAMED[s];
	if (named) return parseHex(named.slice(1));

	const m = /^([a-z]+)\((.*)\)$/.exec(s);
	if (!m) return null;
	const [, fn, body] = m;
	const t = fnArgs(body);
	switch (fn) {
		case "rgb":
		case "rgba":
			return [
				clamp01(num(t[0], 255) / 255),
				clamp01(num(t[1], 255) / 255),
				clamp01(num(t[2], 255) / 255),
				alpha(t[3]),
			];
		case "hsl":
		case "hsla": {
			const [r, g, b] = hslToRgb(
				((hue(t[0]) % 360) + 360) % 360,
				clamp01(num(t[1], 100) / 100),
				clamp01(num(t[2], 100) / 100),
			);
			return [r, g, b, alpha(t[3])];
		}
		case "oklch": {
			const L = num(t[0], 1);
			const C = num(t[1], 0.4);
			const H = (hue(t[2]) * Math.PI) / 180;
			const [r, g, b] = oklabToRgb(L, C * Math.cos(H), C * Math.sin(H));
			return [r, g, b, alpha(t[3])];
		}
		case "oklab": {
			const [r, g, b] = oklabToRgb(
				num(t[0], 1),
				num(t[1], 0.4),
				num(t[2], 0.4),
			);
			return [r, g, b, alpha(t[3])];
		}
		default:
			return null;
	}
}

/** Formats straight sRGB as `#rrggbb` (or `#rrggbbaa` when translucent). */
export function formatHex([r, g, b, a]: Rgba): string {
	const h = (v: number) =>
		Math.round(clamp01(v) * 255)
			.toString(16)
			.padStart(2, "0");
	return `#${h(r)}${h(g)}${h(b)}${a < 0.999 ? h(a) : ""}`;
}

const parseOr = (c: string): Rgba => parseCssColor(c) ?? [0, 0, 0, 0];

/**
 * Mixes two colors perceptually (in OKLab), `t` = 0 → `a`, 1 → `b`.
 * Midpoints keep their lightness and saturation instead of going muddy
 * the way a plain sRGB lerp does.
 */
export function mixColors(a: string, b: string, t: number): string {
	const ca = parseOr(a);
	const cb = parseOr(b);
	// Transparent ends take the other end's hue, so fades don't go grey.
	const pa = ca[3] < 1e-4 ? cb : ca;
	const pb = cb[3] < 1e-4 ? ca : cb;
	const la = rgbToOklab(pa[0], pa[1], pa[2]);
	const lb = rgbToOklab(pb[0], pb[1], pb[2]);
	const k = clamp01(t);
	const [r, g, bl] = oklabToRgb(
		la[0] + (lb[0] - la[0]) * k,
		la[1] + (lb[1] - la[1]) * k,
		la[2] + (lb[2] - la[2]) * k,
	);
	return formatHex([r, g, bl, ca[3] + (cb[3] - ca[3]) * k]);
}

export interface ColorStopLike {
	offset: number;
	color: string;
}

/** Samples a multi-stop gradient (stops in any order) at `t` ∈ [0, 1]. */
export function sampleColorStops(
	stops: readonly (string | ColorStopLike)[],
	t: number,
): string {
	const list = normalizeColorStops(stops);
	if (list.length === 0) return "#00000000";
	if (t <= list[0].offset) return list[0].color;
	for (let i = 1; i < list.length; i++) {
		const b = list[i];
		if (t <= b.offset) {
			const a = list[i - 1];
			const span = b.offset - a.offset;
			return mixColors(a.color, b.color, span > 0 ? (t - a.offset) / span : 1);
		}
	}
	return list[list.length - 1].color;
}

/** Bare color strings become evenly spaced stops; result is sorted by offset. */
export function normalizeColorStops(
	stops: readonly (string | ColorStopLike)[],
): ColorStopLike[] {
	const n = stops.length;
	return stops
		.map((s, i) =>
			typeof s === "string"
				? { offset: n > 1 ? i / (n - 1) : 0, color: s }
				: { offset: clamp01(s.offset), color: s.color },
		)
		.sort((a, b) => a.offset - b.offset);
}

function oklchOf(c: string): [number, number, number, number] {
	const [r, g, b, a] = parseOr(c);
	const [L, A, B] = rgbToOklab(r, g, b);
	return [L, Math.hypot(A, B), (Math.atan2(B, A) * 180) / Math.PI, a];
}

function fromOklch(L: number, C: number, H: number, a: number): string {
	const h = (H * Math.PI) / 180;
	const [r, g, b] = oklabToRgb(L, C * Math.cos(h), C * Math.sin(h));
	return formatHex([r, g, b, a]);
}

/**
 * Authoring helpers for colors. Every function returns a hex string, so the
 * result drops straight into any color prop or keyframe.
 *
 * ```ts
 * Color.oklch(0.7, 0.15, 250)          // perceptual: equal L reads equally bright
 * Color.hsl(212, 0.95, 0.74)
 * Color.mix("#ff5a1f", "#1f6bff", 0.5) // OKLab, no muddy midpoint
 * const wave = Color.ramp(["#3b82f6", "#8b5cf6", "#ec4899"]);
 * wave(0.5)                            // sample a gradient
 * Color.lighten("#ff5a1f", 0.1), Color.alpha("#000", 0.4)
 * ```
 */
export const Color = {
	/** Parses any supported CSS color to straight sRGB 0..1, or null. */
	parse: parseCssColor,
	/** Normalizes any supported CSS color to hex. */
	hex(c: string): string {
		return formatHex(parseOr(c));
	},
	/** `r, g, b` in 0..255, `a` in 0..1. */
	rgb(r: number, g: number, b: number, a = 1): string {
		return formatHex([r / 255, g / 255, b / 255, a]);
	},
	/** Hue in degrees; saturation and lightness in 0..1. */
	hsl(h: number, s: number, l: number, a = 1): string {
		const [r, g, b] = hslToRgb(((h % 360) + 360) % 360, clamp01(s), clamp01(l));
		return formatHex([r, g, b, a]);
	},
	/**
	 * Perceptual lightness 0..1, chroma ~0..0.37, hue in degrees. Colors with
	 * the same `l` look equally bright, which makes balanced palettes easy.
	 */
	oklch(l: number, c: number, h: number, a = 1): string {
		return fromOklch(l, c, h, a);
	},
	/** Perceptual (OKLab) mix; `t` = 0 → `a`, 1 → `b`. */
	mix: mixColors,
	/** A gradient sampler: `Color.ramp(["#f00", "#00f"])(0.25)`. */
	ramp(stops: readonly (string | ColorStopLike)[]): (t: number) => string {
		const list = normalizeColorStops(stops);
		return (t: number) => sampleColorStops(list, t);
	},
	/** `n` evenly spaced swatches along a gradient. */
	steps(stops: readonly (string | ColorStopLike)[], n: number): string[] {
		const list = normalizeColorStops(stops);
		return Array.from({ length: n }, (_, i) =>
			sampleColorStops(list, n > 1 ? i / (n - 1) : 0),
		);
	},
	/** Raises perceptual lightness by `amount` (0..1). */
	lighten(c: string, amount: number): string {
		const [L, C, H, a] = oklchOf(c);
		return fromOklch(clamp01(L + amount), C, H, a);
	},
	darken(c: string, amount: number): string {
		const [L, C, H, a] = oklchOf(c);
		return fromOklch(clamp01(L - amount), C, H, a);
	},
	/** Scales chroma: 0 = grey, 1 = unchanged, >1 = more vivid. */
	saturate(c: string, factor: number): string {
		const [L, C, H, a] = oklchOf(c);
		return fromOklch(L, C * factor, H, a);
	},
	/** Rotates hue by `degrees`, keeping perceptual lightness. */
	rotate(c: string, degrees: number): string {
		const [L, C, H, a] = oklchOf(c);
		return fromOklch(L, C, H + degrees, a);
	},
	/** Replaces alpha. */
	alpha(c: string, a: number): string {
		const [r, g, b] = parseOr(c);
		return formatHex([r, g, b, clamp01(a)]);
	},
	/** WCAG relative-luminance contrast ratio (1..21). ≥ 4.5 for body text. */
	contrast(a: string, b: string): number {
		const lum = (c: string) => {
			const [r, g, bl] = parseOr(c);
			return (
				0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(bl)
			);
		};
		const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
		return (hi + 0.05) / (lo + 0.05);
	},
	/** Black or white, whichever reads better on `background`. */
	readableOn(background: string): string {
		return Color.contrast(background, "#ffffff") >=
			Color.contrast(background, "#000000")
			? "#ffffff"
			: "#000000";
	},
} as const;
