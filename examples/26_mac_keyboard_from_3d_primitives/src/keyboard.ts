/**
 * The keyboard as 3D primitives: an aluminium `Layer3D.roundedBox` body,
 * one tapered rounded box per keycap (equal sizes share one mesh), flat
 * `Layer3D.text` legends lying on the caps, and a `Layer3D.cylinder` Touch
 * ID ring. World axes: x right, y down, z away from the camera, so the
 * keyboard lies in the x/z plane and "up" is -y.
 */
import {
	type AnimatableNode,
	Layer,
	Layer3D,
	type LayerAnimation,
	type LayoutNode,
} from "framefields";
import {
	KEYBOARD_UNITS_D,
	KEYBOARD_UNITS_W,
	KEYS,
	type KeyDef,
} from "./layout.js";
import { ALUMINIUM, CAP, LEGEND, SANS } from "./theme.js";

/** `.animate(...given(a))` animates only when there is an animation. */
const given = (a: LayerAnimation | undefined) => (a ? [a] : []);

/** One key unit in world px. */
export const U = 78;
/** Space between neighbouring caps. */
const GAP = 13;
const CAP_H = 15;
const BODY_H = 24;
const BODY_MARGIN = 0.32 * U;

export const KEYBOARD_W = KEYBOARD_UNITS_W * U + BODY_MARGIN * 2;
export const KEYBOARD_D = KEYBOARD_UNITS_D * U + BODY_MARGIN * 2;
/** Height of the body's top face and of a resting cap's top face (y is down). */
export const BODY_TOP = -BODY_H / 2;
export const CAP_TOP = BODY_TOP - CAP_H;

/** A key's center in the keyboard's own space. */
export function keyCenter(key: KeyDef): { x: number; z: number } {
	return {
		x: (key.x - KEYBOARD_UNITS_W / 2) * U,
		// Row 0 is the back row: far from the camera, so +z.
		z: (KEYBOARD_UNITS_D / 2 - key.z) * U,
	};
}

export interface KeyAnimations {
	/** Moves the whole key (cap and legends): drop-in, press travel. */
	key?: LayerAnimation;
	/** Animates the cap alone, e.g. its `color` tint for the backlight. */
	cap?: LayerAnimation;
}

export interface KeyboardOptions {
	id: string;
	x: number;
	y: number;
	z: number;
	/** Per-key animation; called once for every key. */
	animateKey?: (key: KeyDef, index: number) => KeyAnimations;
	body?: LayerAnimation;
}

/** A legend lying flat on a cap's top face, reading toward the camera. */
function legend(
	id: string,
	text: string,
	size: number,
	dz: number,
): AnimatableNode<LayoutNode> {
	return Layer3D.text({
		id,
		text,
		fontFamily: SANS,
		fontWeight: 500,
		fontSize: size,
		fill: LEGEND,
		material: "unlit",
		x: 0,
		y: CAP_TOP - 0.6,
		z: dz,
		rotateX: -90,
		twoSided: false,
	});
}

/**
 * Glyphs Inter lacks, drawn as stroked paths (Lucide's command and option
 * icons, 24-unit grid scaled to `k`).
 */
const GLYPH_PATHS: Record<string, (k: number) => string> = {
	"⌘": (k) =>
		`M${15 * k} ${6 * k}v${12 * k}a${3 * k} ${3 * k} 0 1 0 ${3 * k} ${-3 * k}` +
		`H${6 * k}a${3 * k} ${3 * k} 0 1 0 ${3 * k} ${3 * k}V${6 * k}` +
		`a${3 * k} ${3 * k} 0 1 0 ${-3 * k} ${3 * k}h${12 * k}` +
		`a${3 * k} ${3 * k} 0 1 0 ${-3 * k} ${-3 * k}`,
	"⌥": (k) =>
		`M${3 * k} ${3 * k}h${6 * k}l${6 * k} ${18 * k}h${6 * k}` +
		`M${14 * k} ${3 * k}h${7 * k}`,
};

/** A vector glyph on a transparent card lying on the cap, like `legend`. */
function glyphLegend(
	id: string,
	glyph: string,
	size: number,
	dz: number,
): LayoutNode {
	const path = GLYPH_PATHS[glyph]!;
	return Layer3D.plane({
		id,
		width: size,
		height: size,
		background: "transparent",
		x: 0,
		y: CAP_TOP - 0.6,
		z: dz,
		rotateX: -90,
		twoSided: false,
		children: [
			Layer.shape("path", {
				id: `${id}-path`,
				position: "absolute",
				x: 0,
				y: 0,
				width: size,
				height: size,
				d: path(size / 24),
				fillColor: "transparent",
				strokeColor: LEGEND,
				strokeWidth: size / 13,
			}),
		],
	});
}

function keyNode(key: KeyDef, anims: KeyAnimations): LayoutNode {
	const w = key.w * U - GAP;
	const d = key.d * U - GAP;
	const { x, z } = keyCenter(key);
	const children: LayoutNode[] = [
		Layer3D.roundedBox({
			id: `key-${key.id}-cap`,
			width: w,
			height: CAP_H,
			depth: d,
			radius: 6,
			segments: 3,
			taper: 0.95,
			y: BODY_TOP - CAP_H / 2,
			color: key.touchId ? "#1b1c1f" : CAP,
			roughness: 0.55,
			shininess: 24,
			specularIntensity: 0.25,
			...(anims.cap && { tint: "#ffffff" }),
		}).animate(...given(anims.cap)),
	];
	if (key.touchId) {
		children.push(
			Layer3D.cylinder({
				id: `key-${key.id}-sensor`,
				radius: d * 0.36,
				height: 1.2,
				segments: 48,
				y: CAP_TOP - 0.6,
				color: "#2f3136",
				metallic: 0.8,
				roughness: 0.25,
				shininess: 80,
			}),
		);
	}
	const small = 0.15 * U;
	const big = 0.25 * U;
	if (key.sub && key.label) {
		const size = key.small ? small : 0.19 * U;
		children.push(
			GLYPH_PATHS[key.sub]
				? glyphLegend(`key-${key.id}-sub`, key.sub, size * 0.95, d * 0.18)
				: legend(`key-${key.id}-sub`, key.sub, size, d * 0.18),
			legend(`key-${key.id}-label`, key.label, size, -d * 0.18),
		);
	} else if (key.label) {
		children.push(
			legend(`key-${key.id}-label`, key.label, key.small ? small : big, 0),
		);
	}
	return Layer.box({
		id: `key-${key.id}`,
		position: "absolute",
		x,
		y: 0,
		z,
		width: w,
		height: d,
		is3D: true,
		transformStyle: "preserve-3d",
		background: "transparent",
		children,
	}).animate(...given(anims.key)) as unknown as LayoutNode;
}

export function keyboard(options: KeyboardOptions): AnimatableNode<LayoutNode> {
	const body = Layer3D.roundedBox({
		id: `${options.id}-body`,
		width: KEYBOARD_W,
		height: BODY_H,
		depth: KEYBOARD_D,
		radius: 11,
		segments: 5,
		color: ALUMINIUM,
		metallic: 0.55,
		roughness: 0.38,
		shininess: 60,
		specularIntensity: 0.7,
	}).animate(...given(options.body));
	const keys = KEYS.map((key, i) =>
		keyNode(key, options.animateKey?.(key, i) ?? {}),
	);
	return Layer.box({
		id: options.id,
		position: "absolute",
		x: options.x,
		y: options.y,
		z: options.z,
		width: KEYBOARD_W,
		height: KEYBOARD_D,
		is3D: true,
		transformStyle: "preserve-3d",
		background: "transparent",
		children: [body, ...keys],
	}) as unknown as AnimatableNode<LayoutNode>;
}
