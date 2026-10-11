// Pixel helpers for render tests (decoding PNG frames and comparing them).
import { Canvas, loadImage } from "skia-canvas";

export interface DecodedImage {
	width: number;
	height: number;
	pixels: Uint8ClampedArray;
}

export interface PixelRgba {
	r: number;
	g: number;
	b: number;
	a: number;
}

export async function decodePngPixels(
	pngBuffer: Buffer,
): Promise<DecodedImage> {
	const img = await loadImage(pngBuffer);
	const canvas = new Canvas(img.width, img.height);
	const ctx = canvas.getContext("2d");
	ctx.drawImage(img, 0, 0);
	const imgData = ctx.getImageData(0, 0, img.width, img.height);
	return {
		width: img.width,
		height: img.height,
		pixels: imgData.data,
	};
}

export function getPixel(image: DecodedImage, x: number, y: number): PixelRgba {
	const clampedX = Math.max(0, Math.min(image.width - 1, Math.round(x)));
	const clampedY = Math.max(0, Math.min(image.height - 1, Math.round(y)));
	const index = (clampedY * image.width + clampedX) * 4;
	return {
		r: image.pixels[index] ?? 0,
		g: image.pixels[index + 1] ?? 0,
		b: image.pixels[index + 2] ?? 0,
		a: image.pixels[index + 3] ?? 0,
	};
}

export function computeMse(
	pixelsA: Uint8ClampedArray,
	pixelsB: Uint8ClampedArray,
): number {
	if (pixelsA.length !== pixelsB.length) {
		throw new Error(
			`Pixel buffer size mismatch: ${pixelsA.length} vs ${pixelsB.length}`,
		);
	}
	let sumSquaredDiff = 0;
	for (let i = 0; i < pixelsA.length; i += 4) {
		const dr = (pixelsA[i] ?? 0) - (pixelsB[i] ?? 0);
		const dg = (pixelsA[i + 1] ?? 0) - (pixelsB[i + 1] ?? 0);
		const db = (pixelsA[i + 2] ?? 0) - (pixelsB[i + 2] ?? 0);
		sumSquaredDiff += (dr * dr + dg * dg + db * db) / 3;
	}
	return sumSquaredDiff / (pixelsA.length / 4);
}
