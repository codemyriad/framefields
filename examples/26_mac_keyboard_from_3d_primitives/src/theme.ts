import path from "node:path";
import { fileURLToPath } from "node:url";
import { FontManager } from "framefields";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FONTS = path.resolve(HERE, "../../../assets/fonts");
export const OUTPUT = path.resolve(HERE, "../output");

export const W = 1920;
export const H = 1080;
export const FPS = 30;

/** Studio: a near-black sweep, so the white caps and silver body read. It
 * matches the floor where it meets the far plane, so there is no horizon. */
export const BG = "#101114";
export const TEXT = "#f3f4f6";
export const MUTED = "#8b8f98";
/** Backlight: what a pressed key glows, a cool Apple blue. */
export const GLOW = "#9cc3ff";
export const ACCENT = "#5ea0ff";

/** Silver aluminium body and white caps with graphite legends. */
export const ALUMINIUM = "#b4b8be";
export const CAP = "#f4f4f2";
export const LEGEND = "#2b2d31";

export const SANS = "Inter";
export const MONO = "JetBrains Mono";

export async function registerFonts(): Promise<void> {
	await FontManager.register({
		family: SANS,
		source: path.join(FONTS, "Inter.ttf"),
	});
	await FontManager.register({
		family: MONO,
		source: path.join(FONTS, "JetBrainsMono.ttf"),
	});
}
