/**
 * A compact US Magic Keyboard (with Touch ID), in key units: one unit is
 * the pitch of a letter key. Rows run from the function row (back, row 0)
 * to the modifier row (front, row 5); every main row is 15 units wide.
 */
export interface KeyDef {
	/** Stable id: also what `type()` looks keys up by. */
	id: string;
	/** Main legend. */
	label: string;
	/** Second legend above the main one (shifted symbol, modifier glyph). */
	sub?: string;
	/** Center, in units from the keyboard's left edge. */
	x: number;
	/** Center, in units from the back edge. */
	z: number;
	/** Width and depth in units. */
	w: number;
	d: number;
	/** Small legends: words on modifiers, F-keys. */
	small?: boolean;
	/** The Touch ID button: a dark cap with a sensor ring. */
	touchId?: boolean;
}

export const KEYBOARD_UNITS_W = 15;
const FN_ROW_D = 0.6;
export const KEYBOARD_UNITS_D = FN_ROW_D + 5;

type RowItem = [id: string, label: string, w?: number, sub?: string];

const ROWS: RowItem[][] = [
	[
		["esc", "esc", 1.5],
		...Array.from(
			{ length: 12 },
			(_, i) => [`f${i + 1}`, `F${i + 1}`, 1] as RowItem,
		),
		["touchid", "", 1.5],
	],
	[
		["grave", "`", 1, "~"],
		["1", "1", 1, "!"],
		["2", "2", 1, "@"],
		["3", "3", 1, "#"],
		["4", "4", 1, "$"],
		["5", "5", 1, "%"],
		["6", "6", 1, "^"],
		["7", "7", 1, "&"],
		["8", "8", 1, "*"],
		["9", "9", 1, "("],
		["0", "0", 1, ")"],
		["minus", "-", 1, "_"],
		["equal", "=", 1, "+"],
		["delete", "delete", 2],
	],
	[
		["tab", "tab", 1.5],
		..."QWERTYUIOP".split("").map((c) => [c.toLowerCase(), c] as RowItem),
		["lbracket", "[", 1, "{"],
		["rbracket", "]", 1, "}"],
		["backslash", "\\", 1.5, "|"],
	],
	[
		["caps", "caps lock", 1.75],
		..."ASDFGHJKL".split("").map((c) => [c.toLowerCase(), c] as RowItem),
		["semicolon", ";", 1, ":"],
		["quote", "'", 1, '"'],
		["return", "return", 2.25],
	],
	[
		["lshift", "shift", 2.25, "⇧"],
		..."ZXCVBNM".split("").map((c) => [c.toLowerCase(), c] as RowItem),
		["comma", ",", 1, "<"],
		["period", ".", 1, ">"],
		["slash", "/", 1, "?"],
		["rshift", "shift", 2.75, "⇧"],
	],
	[
		["fn", "fn", 1],
		["lctrl", "control", 1, "^"],
		["lopt", "option", 1, "⌥"],
		["lcmd", "command", 1.25, "⌘"],
		["space", "", 5.5],
		["rcmd", "command", 1.25, "⌘"],
		["ropt", "option", 1, "⌥"],
		// The arrow cluster is laid out by hand below.
	],
];

/** Labels set in the small size. */
const SMALL =
	/^(esc|F\d+|delete|tab|caps lock|return|shift|fn|control|option|command)$/;

function buildLayout(): KeyDef[] {
	const keys: KeyDef[] = [];
	ROWS.forEach((row, r) => {
		const d = r === 0 ? FN_ROW_D : 1;
		const z = r === 0 ? FN_ROW_D / 2 : FN_ROW_D + (r - 1) + 0.5;
		let x = 0;
		for (const [id, label, w = 1, sub] of row) {
			keys.push({
				id,
				label,
				sub,
				x: x + w / 2,
				z,
				w,
				d,
				small: SMALL.test(label),
				touchId: id === "touchid",
			});
			x += w;
		}
	});
	// Arrows: ← and → are half-height, ↑ and ↓ share one key slot.
	const rowZ = FN_ROW_D + 4;
	const arrows: Array<[string, string, number, number]> = [
		["left", "←", 12.5, rowZ + 0.75],
		["up", "↑", 13.5, rowZ + 0.25],
		["down", "↓", 13.5, rowZ + 0.75],
		["right", "→", 14.5, rowZ + 0.75],
	];
	for (const [id, label, x, z] of arrows) {
		keys.push({ id, label, x, z, w: 1, d: 0.5, small: true });
	}
	return keys;
}

export const KEYS: readonly KeyDef[] = buildLayout();

/** Which keys a character needs: its own key, plus shift for capitals and symbols. */
export function keysForChar(ch: string): string[] {
	if (ch === " ") return ["space"];
	if (ch === "\n") return ["return"];
	const lower = ch.toLowerCase();
	const letter = KEYS.find((k) => k.id === lower && /^[a-z0-9]$/.test(k.id));
	if (letter) return ch !== lower ? ["lshift", letter.id] : [letter.id];
	const plain = KEYS.find((k) => k.label === ch);
	if (plain) return [plain.id];
	const shifted = KEYS.find((k) => k.sub === ch);
	if (shifted) return ["lshift", shifted.id];
	throw new Error(`No key types ${JSON.stringify(ch)}`);
}
