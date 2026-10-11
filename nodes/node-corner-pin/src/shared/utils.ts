import { solveHomography } from "@framefields/node-sdk";

export function getDefaultPoints(): { x: number; y: number }[] {
	return [
		{ x: 0, y: 0 }, // Top-Left (TL)
		{ x: 100, y: 0 }, // Top-Right (TR)
		{ x: 0, y: 100 }, // Bottom-Left (BL)
		{ x: 100, y: 100 }, // Bottom-Right (BR)
	];
}

/**
 * Homography mapping the corner-pin quad (percent coordinates, TL, TR, BL, BR)
 * to the unit square it samples from. Falls back to the identity quad.
 */
export function solveCornerPinHomography(
	targetPoints: { x: number; y: number }[],
): number[] {
	const pts =
		targetPoints && targetPoints.length === 4
			? targetPoints
			: getDefaultPoints();
	return solveHomography(pts.map((p) => ({ x: p.x / 100.0, y: p.y / 100.0 })));
}
