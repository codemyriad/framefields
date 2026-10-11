/**
 * Scene sequencing with transitions. `Layer.sequence` lays scenes end to end
 * and overlaps neighbours by each transition's length; the transition itself
 * is plain keyframes (and, for wipes and irises, a track matte) merged into
 * the scenes, so it renders like anything else and stays editable.
 */
import {
	type AnimationTrack,
	type BoxNode,
	LayerAnimation,
	type LayoutNode,
} from "@framefields/compositions/program";
import { autoId } from "./ids.js";
import { type AnimatableNode, Layer } from "./index.js";

export type TransitionDirection = "left" | "right" | "up" | "down";

/** What a transition needs to know to animate two scenes. */
interface TransitionContext {
	/** Scene size (px). */
	width: number;
	height: number;
	/** Overlap length in frames. */
	frames: number;
	/** Outgoing scene: local frame where the overlap starts, and its rest position. */
	outStart: number;
	outRest: { x: number; y: number };
	/** Incoming scene rest position (the overlap starts at its local frame 0). */
	inRest: { x: number; y: number };
}

interface TransitionParts {
	outgoing?: LayerAnimation;
	incoming?: LayerAnimation;
	/** A matte revealing the incoming scene (laid out like the scene, keyed from its start). */
	incomingMatte?: LayoutNode;
	/** Static props set on both scenes (e.g. motion blur for a whip). */
	sceneProps?: Record<string, unknown>;
}

export interface TransitionSpec {
	readonly kind: string;
	/** Overlap between the two scenes, in frames (0 = hard cut). */
	readonly frames: number;
	/** @internal */
	build(ctx: TransitionContext): TransitionParts;
}

const offsetFor = (dir: TransitionDirection, w: number, h: number) => {
	switch (dir) {
		case "left":
			return { dx: -w, dy: 0 };
		case "right":
			return { dx: w, dy: 0 };
		case "up":
			return { dx: 0, dy: -h };
		default:
			return { dx: 0, dy: h };
	}
};

function spec(
	kind: string,
	frames: number,
	build: (ctx: TransitionContext) => TransitionParts,
): TransitionSpec {
	return { kind, frames: Math.max(0, Math.round(frames)), build };
}

/**
 * Scene-to-scene transitions for `Layer.sequence`. `frames` is how long the
 * two scenes overlap; `ease` shapes the move (default "power3.inOut").
 */
export const Transition = {
	/** Instant cut. */
	cut: (): TransitionSpec => spec("cut", 0, () => ({})),

	/** The incoming scene fades in over the outgoing one. */
	crossfade: (frames = 15, ease = "sine.inOut"): TransitionSpec =>
		spec("crossfade", frames, (c) => ({
			incoming: LayerAnimation.keys("opacity", [
				[0, 0],
				[c.frames, 1, ease],
			]),
		})),

	/** Out to the background, then in: a dip to the composition's background color. */
	fadeThrough: (frames = 20, ease = "sine.inOut"): TransitionSpec =>
		spec("fadeThrough", frames, (c) => {
			const half = Math.max(1, Math.floor(c.frames / 2));
			return {
				outgoing: LayerAnimation.keys("opacity", [
					[c.outStart, 1],
					[c.outStart + half, 0, ease],
				]),
				incoming: LayerAnimation.keys("opacity", [
					[0, 0],
					[half, 0],
					[c.frames, 1, ease],
				]),
			};
		}),

	/** Both scenes move together: the new one pushes the old one out toward `direction`. */
	push: (
		direction: TransitionDirection = "left",
		frames = 18,
		ease = "power3.inOut",
	): TransitionSpec =>
		spec("push", frames, (c) => {
			const { dx, dy } = offsetFor(direction, c.width, c.height);
			const out = LayerAnimation.create();
			const inc = LayerAnimation.create();
			if (dx !== 0) {
				out.keys("x", [
					[c.outStart, c.outRest.x],
					[c.outStart + c.frames, c.outRest.x + dx, ease],
				]);
				inc.keys("x", [
					[0, c.inRest.x - dx],
					[c.frames, c.inRest.x, ease],
				]);
			} else {
				out.keys("y", [
					[c.outStart, c.outRest.y],
					[c.outStart + c.frames, c.outRest.y + dy, ease],
				]);
				inc.keys("y", [
					[0, c.inRest.y - dy],
					[c.frames, c.inRest.y, ease],
				]);
			}
			return { outgoing: out, incoming: inc };
		}),

	/** The new scene slides in over the old one, which stays put and dims slightly. */
	slide: (
		direction: TransitionDirection = "left",
		frames = 18,
		ease = "power3.out",
	): TransitionSpec =>
		spec("slide", frames, (c) => {
			const { dx, dy } = offsetFor(direction, c.width, c.height);
			const prop = dx !== 0 ? "x" : "y";
			const rest = dx !== 0 ? c.inRest.x : c.inRest.y;
			return {
				outgoing: LayerAnimation.keys("opacity", [
					[c.outStart, 1],
					[c.outStart + c.frames, 0.6, ease],
				]),
				incoming: LayerAnimation.keys(prop, [
					[0, rest - (dx || dy)],
					[c.frames, rest, ease],
				]),
			};
		}),

	/** Old scene grows and fades, new one settles in from slightly small. */
	zoom: (frames = 18, ease = "power2.inOut"): TransitionSpec =>
		spec("zoom", frames, (c) => ({
			outgoing: LayerAnimation.create()
				.keys("scale", [
					[c.outStart, 1],
					[c.outStart + c.frames, 1.25, ease],
				])
				.keys("opacity", [
					[c.outStart, 1],
					[c.outStart + c.frames, 0, ease],
				]),
			incoming: LayerAnimation.create()
				.keys("scale", [
					[0, 0.85],
					[c.frames, 1, ease],
				])
				.keys("opacity", [
					[0, 0],
					[c.frames, 1, ease],
				]),
		})),

	/** A fast, motion-blurred push: the camera whips to the next scene. */
	whip: (
		direction: TransitionDirection = "left",
		frames = 10,
	): TransitionSpec => {
		const push = Transition.push(direction, frames, "expo.inOut");
		return spec("whip", frames, (c) => ({
			...push.build(c),
			sceneProps: { motionBlurShutter: 360 },
		}));
	},

	/** A hard edge sweeps across, revealing the new scene. */
	wipe: (
		direction: TransitionDirection = "left",
		frames = 18,
		ease = "power2.inOut",
	): TransitionSpec =>
		spec("wipe", frames, (c) => {
			// `direction` is where the edge travels: "left" reveals from the right.
			const { dx, dy } = offsetFor(direction, c.width, c.height);
			const prop = dx !== 0 ? "x" : "y";
			const rest = dx !== 0 ? c.inRest.x : c.inRest.y;
			return {
				incomingMatte: Layer.box({
					id: autoId("wipe-matte"),
					position: "absolute",
					x: c.inRest.x,
					y: c.inRest.y,
					width: c.width,
					height: c.height,
					background: "#ffffff",
				}).animate(
					LayerAnimation.keys(prop, [
						[0, rest - (dx || dy)],
						[c.frames, rest, ease],
					]),
				),
			};
		}),

	/** A circle opens from the center (or `at`, as fractions of the frame) to reveal the new scene. */
	iris: (
		frames = 20,
		options: { at?: [number, number]; ease?: string } = {},
	): TransitionSpec =>
		spec("iris", frames, (c) => {
			const [fx, fy] = options.at ?? [0.5, 0.5];
			const cx = c.inRest.x + fx * c.width;
			const cy = c.inRest.y + fy * c.height;
			// Big enough to cover the farthest corner.
			const r = Math.hypot(
				Math.max(fx, 1 - fx) * c.width,
				Math.max(fy, 1 - fy) * c.height,
			);
			return {
				incomingMatte: Layer.shape("circle", {
					id: autoId("iris-matte"),
					position: "absolute",
					x: cx - r,
					y: cy - r,
					width: r * 2,
					height: r * 2,
					fillColor: "#ffffff",
				}).animate(
					LayerAnimation.keys("scale", [
						[0, 0],
						[c.frames, 1, options.ease ?? "power2.inOut"],
					]),
				),
			};
		}),
};

export interface SequenceEntry {
	/** A full-frame layer (usually a box holding the scene). */
	scene: AnimatableNode<LayoutNode> | LayoutNode;
	/** Frames this scene is on screen, transitions included. */
	duration: number;
	/** How this scene arrives from the previous one (default: cut). */
	transition?: TransitionSpec;
}

export interface SceneSequenceOptions {
	id?: string;
	/** Scene size used by moving transitions (default: each scene's own width/height, else 1920×1080). */
	width?: number;
	height?: number;
	/** First frame of the whole sequence. Default 0. */
	startFrame?: number;
}

export type SequenceNode = AnimatableNode<BoxNode> & {
	/** Total length in frames: put it in the composition's duration. */
	readonly totalFrames: number;
	/** Where each scene starts (composition frames, relative to the sequence). */
	readonly sceneStarts: readonly number[];
};

function sizeOf(node: LayoutNode, axis: "width" | "height", fallback: number) {
	const v = (node as unknown as Record<string, unknown>)[axis];
	return typeof v === "number" ? v : fallback;
}

/**
 * Scenes back to back, each overlapping the previous one by its transition.
 * Scene timing is set for you (`startFrame` / `durationFrames`), so every
 * scene keys its own animation from its local frame 0.
 */
export function sequence(
	entries: SequenceEntry[],
	options: SceneSequenceOptions = {},
): SequenceNode {
	const scenes: LayoutNode[] = [];
	const starts: number[] = [];
	let cursor = 0;
	entries.forEach((entry, i) => {
		const t = i === 0 ? undefined : entry.transition;
		const overlap = Math.min(t?.frames ?? 0, entries[i - 1]?.duration ?? 0);
		const start = Math.max(0, cursor - overlap);
		const duration = Math.max(1, Math.round(entry.duration));
		starts.push(start);
		const node = entry.scene as LayoutNode & Record<string, unknown>;
		// Scenes stack in one frame rather than flowing one under the other.
		if (node.position === undefined || node.position === "relative") {
			node.position = "absolute";
			node.x ??= 0;
			node.y ??= 0;
		}
		node.startFrame = start;
		node.durationFrames = duration;
		scenes.push(node);
		cursor = start + duration;
	});

	entries.forEach((entry, i) => {
		if (i === 0 || !entry.transition || entry.transition.frames === 0) return;
		const prev = scenes[i - 1] as LayoutNode & Record<string, unknown>;
		const cur = scenes[i] as LayoutNode & Record<string, unknown>;
		const frames = Math.min(entry.transition.frames, entries[i - 1].duration);
		const width = options.width ?? sizeOf(cur, "width", 1920);
		const height = options.height ?? sizeOf(cur, "height", 1080);
		const parts = entry.transition.build({
			width,
			height,
			frames,
			outStart: entries[i - 1].duration - frames,
			outRest: {
				x: typeof prev.x === "number" ? prev.x : 0,
				y: typeof prev.y === "number" ? prev.y : 0,
			},
			inRest: {
				x: typeof cur.x === "number" ? cur.x : 0,
				y: typeof cur.y === "number" ? cur.y : 0,
			},
		});
		const merge = (
			node: LayoutNode & Record<string, unknown>,
			anim?: LayerAnimation,
		) => {
			if (!anim) return;
			const existing =
				(node.animation as { tracks?: AnimationTrack[] } | undefined)?.tracks ??
				[];
			node.animation = LayerAnimation.merge(
				{ tracks: existing },
				anim,
			).toSpec();
		};
		merge(prev, parts.outgoing);
		merge(cur, parts.incoming);
		if (parts.sceneProps) {
			Object.assign(prev, parts.sceneProps);
			Object.assign(cur, parts.sceneProps);
		}
		if (parts.incomingMatte) {
			cur.matte = { node: parts.incomingMatte, mode: "alpha" };
		}
	});

	const root = Layer.box({
		id: options.id ?? autoId("sequence"),
		position: "absolute",
		x: 0,
		y: 0,
		width:
			options.width ?? sizeOf(scenes[0] ?? ({} as LayoutNode), "width", 1920),
		height:
			options.height ?? sizeOf(scenes[0] ?? ({} as LayoutNode), "height", 1080),
		startFrame: options.startFrame ?? 0,
		durationFrames: Math.max(1, cursor),
		children: scenes,
	});
	Object.defineProperty(root, "totalFrames", {
		value: cursor,
		enumerable: false,
	});
	Object.defineProperty(root, "sceneStarts", {
		value: starts,
		enumerable: false,
	});
	return root as SequenceNode;
}
