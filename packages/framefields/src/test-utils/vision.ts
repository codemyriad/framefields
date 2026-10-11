import { Layer, VisionNode } from "../index.js";

/**
 * A video with person detection whose frame-0 result is one synthetic person:
 * a 200×500 box at (400, 300) in a 1920×1080 frame.
 */
export function attachSyntheticPersonTrack(src = "test.mp4") {
	const video = Layer.video(src);
	const vision = VisionNode.attach(video, {
		enableDetection: true,
		classes: ["person"],
	});
	vision.setObjectResult(0, [
		{
			trackId: 1,
			category: "person",
			score: 0.95,
			boundingBox: {
				originX: 400,
				originY: 300,
				width: 200,
				height: 500,
				normalizedX: 400 / 1920,
				normalizedY: 300 / 1080,
				normalizedWidth: 200 / 1920,
				normalizedHeight: 500 / 1080,
			},
			centerX: 500,
			centerY: 550,
			normalizedCenterX: 500 / 1920,
			normalizedCenterY: 550 / 1080,
			velocity: { vx: 0, vy: 0 },
			speed: 0,
			age: 1,
			hits: 1,
			active: true,
			isCoasting: false,
		},
	]);
	return { video, vision };
}
