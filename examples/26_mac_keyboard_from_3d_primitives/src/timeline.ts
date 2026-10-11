import { keysForChar } from "./layout.js";

/** 12 seconds at 30 fps. */
export const DURATION = 360;

/** Act 1: the body lands and the caps rain in from the center out. */
export const ASSEMBLE = { body: 0, keysFrom: 8, keysSpread: 52, land: 16 };

/** Act 2: the line that gets typed, and when typing starts. */
export const LINE = "Hello, framefields.";
export const TYPE_FROM = 112;

/** Act 3: a backlight wave sweeps left to right, then the title. */
export const WAVE_FROM = 262;
export const WAVE_SPREAD = 44;
export const TITLE_AT = 300;

export interface Stroke {
	/** Frame the key bottoms out. */
	at: number;
	/** Key ids pressed together: a modifier first, then the key. */
	keys: string[];
	/** Characters on screen once this stroke lands. */
	count: number;
}

/** Deterministic jitter in [-1, 1], so every render types with the same rhythm. */
function jitter(i: number): number {
	const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
	return (x - Math.floor(x)) * 2 - 1;
}

/** A human rhythm: ~5.5 frames a key, slower on spaces, a breath after punctuation. */
function schedule(): Stroke[] {
	const strokes: Stroke[] = [];
	let at = TYPE_FROM;
	[...LINE].forEach((ch, i) => {
		strokes.push({ at, keys: keysForChar(ch), count: i + 1 });
		let gap = 5.5 + jitter(i) * 1.6;
		if (ch === " ") gap += 2;
		if (ch === ",") gap += 7;
		at += Math.max(3, Math.round(gap));
	});
	strokes.push({ at: at + 10, keys: ["return"], count: LINE.length });
	return strokes;
}

export const STROKES: readonly Stroke[] = schedule();
export const TYPE_END = STROKES[STROKES.length - 1]!.at;
