/**
 * Every key's whole life as one pair of key tracks: it rains in, gets
 * pressed whenever the typing schedule needs it, then rides the backlight
 * wave. Tracks must run forward in time, so each key's moments are
 * collected first and written out in order.
 */
import { LayerAnimation } from "framefields";
import { type KeyAnimations, keyCenter, U } from "./keyboard.js";
import { KEYBOARD_UNITS_W, KEYS, type KeyDef } from "./layout.js";
import { GLOW } from "./theme.js";
import { ASSEMBLE, STROKES, WAVE_FROM, WAVE_SPREAD } from "./timeline.js";

type Key = [frame: number, value: number | string, ease?: string];

/** How far a cap travels when pressed (y is down). */
const TRAVEL = 6;
const DROP = -340;
const REST = "#ffffff";

/** Frames a key is held down: [bottomed out, released]. */
function pressIntervals(id: string): Array<[number, number]> {
	const raw: Array<[number, number]> = [];
	for (const s of STROKES) {
		const i = s.keys.indexOf(id);
		if (i < 0) continue;
		// A modifier goes down first and lets go after its partner.
		const isModifier = i < s.keys.length - 1;
		raw.push(isModifier ? [s.at - 4, s.at + 3] : [s.at, s.at + 1]);
	}
	// Strokes too close to come back up in between merge into one hold.
	const merged: Array<[number, number]> = [];
	for (const iv of raw) {
		const last = merged[merged.length - 1];
		if (last && iv[0] - last[1] < 5) last[1] = Math.max(last[1], iv[1]);
		else merged.push([...iv]);
	}
	return merged;
}

/** Keys in time order; where two land on one frame the later-written wins. */
function ordered(keys: Key[]): Key[] {
	const byFrame = new Map<number, Key>();
	for (const k of keys) byFrame.set(k[0], k);
	return [...byFrame.values()].sort((a, b) => a[0] - b[0]);
}

/** hsl → hex, for the wave's gradient. */
function hsl(h: number, s: number, l: number): string {
	const a = s * Math.min(l, 1 - l);
	const f = (n: number) => {
		const k = (n + h / 30) % 12;
		const c = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
		return Math.round(c * 255)
			.toString(16)
			.padStart(2, "0");
	};
	return `#${f(0)}${f(8)}${f(4)}`;
}

const HALF_W = (KEYBOARD_UNITS_W / 2) * U;
const MAX_DIST = Math.max(
	...KEYS.map((k) => Math.hypot(keyCenter(k).x, keyCenter(k).z)),
);

export function animateKey(key: KeyDef): KeyAnimations {
	const { x, z } = keyCenter(key);
	const y: Key[] = [];
	const color: Key[] = [[0, REST]];

	// Act 1: rain in, nearest the center first.
	const d =
		ASSEMBLE.keysFrom +
		Math.round((Math.hypot(x, z) / MAX_DIST) * ASSEMBLE.keysSpread);
	y.push([0, DROP], [d, DROP], [d + ASSEMBLE.land, 0, "back.out(1.5)"]);
	const opacity: Key[] = [
		[0, 0],
		[d, 0],
		[d + 5, 1, "power1.out"],
	];

	// Act 2: presses, each with a backlight flash that fades after release.
	// A quick repeat (the "ll" in Hello) springs back faster and flashes again.
	const presses = pressIntervals(key.id);
	presses.forEach(([down, up], i) => {
		const next = presses[i + 1]?.[0] ?? Number.POSITIVE_INFINITY;
		const back = Math.min(up + 5, next - 2);
		const fade = Math.min(up + 14, next - 1);
		y.push([down - 2, 0], [down, TRAVEL, "power2.in"]);
		if (up > down) y.push([up, TRAVEL]);
		y.push([back, 0, "power2.out"]);
		if (i === 0) color.push([down - 1, REST]);
		color.push([down, GLOW, "power2.out"]);
		if (up > down) color.push([up, GLOW]);
		color.push([fade, REST, "sine.inOut"]);
	});

	// Act 3: a crest rolls left to right; the gradient runs blue → violet → pink.
	const u = (x + HALF_W) / (2 * HALF_W);
	const w = WAVE_FROM + Math.round(u * WAVE_SPREAD);
	y.push([w, 0], [w + 9, -18, "sine.out"], [w + 22, 0, "sine.inOut"]);
	color.push(
		[w, REST],
		[w + 6, hsl(212 + u * 118, 0.95, 0.74), "power1.out"],
		[w + 40, REST, "sine.inOut"],
	);

	return {
		key: LayerAnimation.create().keys("y", ordered(y)).keys("opacity", opacity),
		cap: LayerAnimation.create().keys("color", ordered(color)),
	};
}
