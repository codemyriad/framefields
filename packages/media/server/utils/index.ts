import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { generateId } from "@framefields/core";
import { mediaLogger } from "@framefields/server-utils";
import { fileTypeFromBuffer } from "file-type";
import sharp from "sharp";

interface ProbeInput {
	inputPath: string;
	/** File type detected from a buffer input. */
	type?: { ext: string; mime: string };
	/** Deletes the temp file a buffer input was written to. */
	cleanup: (logMessage?: string) => Promise<void>;
}

/**
 * A path ffprobe can read: URLs as-is, buffers written to a temp file named
 * `${prefix}_<id><ext>` (ext from the detected type, else `fallbackExt`).
 */
async function prepareProbeInput(
	bufferOrUrl: Buffer | string,
	prefix: string,
	fallbackExt?: (type: { ext: string; mime: string } | undefined) => string,
): Promise<ProbeInput> {
	if (typeof bufferOrUrl === "string") {
		return { inputPath: bufferOrUrl, cleanup: async () => {} };
	}
	const type = await fileTypeFromBuffer(bufferOrUrl);
	const ext = type ? `.${type.ext}` : (fallbackExt?.(type) ?? "");
	const tempFile = path.join(os.tmpdir(), `${prefix}_${generateId()}${ext}`);
	await fs.writeFile(tempFile, bufferOrUrl);
	return {
		inputPath: tempFile,
		type,
		cleanup: async (logMessage = "Failed to delete temp file") => {
			try {
				await fs.unlink(tempFile);
			} catch (err) {
				mediaLogger.error({ err }, logMessage);
			}
		},
	};
}

/** Runs ffprobe to completion; rejects only when it can't be started. */
function runFfprobe(
	args: string[],
): Promise<{ code: number | null; stdout: string; stderr: string }> {
	return new Promise((resolve, reject) => {
		const ffprobe = spawn("ffprobe", args);
		let stdout = "";
		let stderr = "";
		ffprobe.stdout.on("data", (data) => {
			stdout += data.toString();
		});
		ffprobe.stderr.on("data", (data) => {
			stderr += data.toString();
		});
		ffprobe.on("error", (err) => {
			reject(new Error(`Failed to start ffprobe: ${err.message}`));
		});
		ffprobe.on("close", (code) => {
			resolve({ code, stdout, stderr });
		});
	});
}

const MIME_EXTENSIONS: Record<string, string> = {
	"audio/mpeg": ".mp3",
	"audio/mp3": ".mp3",
	"audio/wav": ".wav",
	"audio/x-wav": ".wav",
	"audio/ogg": ".ogg",
	"audio/aac": ".aac",
	"audio/m4a": ".m4a",
	"audio/x-m4a": ".m4a",
	"audio/mp4": ".m4a",
	"video/mp4": ".mp4",
	"video/quicktime": ".mov",
	"video/x-matroska": ".mkv",
};

/**
 * Gets the duration of media (video/audio) using ffprobe.
 */
export async function getMediaDuration(
	bufferOrUrl: Buffer | string,
	mimeType?: string,
): Promise<number | null> {
	// Extension from the detected type, otherwise from the mimeType hint
	const { inputPath, type, cleanup } = await prepareProbeInput(
		bufferOrUrl,
		"temp_media",
		() => (mimeType && MIME_EXTENSIONS[mimeType]) || "",
	);

	try {
		const { code, stdout, stderr } = await runFfprobe([
			"-v",
			"error",
			"-show_entries",
			"format=duration",
			"-of",
			"default=noprint_wrappers=1:nokey=1",
			inputPath,
		]);
		if (code !== 0) {
			const details =
				typeof bufferOrUrl === "string"
					? `url: ${bufferOrUrl}`
					: `buffer size: ${bufferOrUrl.length}, hint: ${mimeType}, detected: ${type?.mime}`;
			throw new Error(
				`ffprobe exited with code ${code}. stderr: ${stderr.trim()}. details: ${details}`,
			);
		}
		const duration = parseFloat(stdout.trim());
		return Number.isNaN(duration) ? null : duration;
	} finally {
		await cleanup();
	}
}
/**
 * Gets video metadata (width, height, fps, duration) using ffprobe.
 */
export async function getVideoMetadata(bufferOrUrl: Buffer | string): Promise<{
	width: number;
	height: number;
	fps: number;
	duration: number;
} | null> {
	const { inputPath, cleanup } = await prepareProbeInput(
		bufferOrUrl,
		"temp_video_meta",
	);

	try {
		const { code, stdout, stderr } = await runFfprobe([
			"-v",
			"error",
			"-select_streams",
			"v:0",
			"-show_entries",
			"stream=width,height,avg_frame_rate,duration",
			"-of",
			"json",
			inputPath,
		]);
		if (code !== 0) {
			throw new Error(
				`ffprobe exited with code ${code}. stderr: ${stderr.trim()}`,
			);
		}
		try {
			const data = JSON.parse(stdout);
			const stream = data.streams?.[0];

			if (!stream) {
				return null;
			}

			const width = Number(stream.width);
			const height = Number(stream.height);
			const duration = parseFloat(stream.duration);

			// Parse avg_frame_rate (e.g., "30/1" or "24000/1001").
			// Chromium MediaRecorder WebM tracks often lack a
			// DefaultDuration element; ffprobe then reports 1000/1
			// (a ~1ms-per-timestamp fallback) even though the capture
			// is real 30/60fps. Guard against absurd values.
			let fps = 0;
			if (stream.avg_frame_rate) {
				const [num, den] = stream.avg_frame_rate.split("/");
				if (num && den) {
					fps = Number(num) / Number(den);
				}
			}
			// Never trust frame rates above 120fps or below 1fps from
			// container heuristics — treat them as unknown so we fall
			// back to a real measurement path.
			if (Number.isFinite(fps) && (fps < 1 || fps > 120)) fps = 0;

			return {
				width: Number.isNaN(width) ? 0 : width,
				height: Number.isNaN(height) ? 0 : height,
				fps: Number.isNaN(fps) ? 0 : fps,
				duration: Number.isNaN(duration) ? 0 : duration,
			};
		} catch (err) {
			throw new Error(`Failed to parse ffprobe output: ${err}`);
		}
	} finally {
		await cleanup();
	}
}
/**
 * Generates a thumbnail buffer from a video URL using FFmpeg and Sharp.
 */
export async function generateVideoThumbnail(
	videoUrl: string,
	width: number,
	height: number,
): Promise<Buffer> {
	return new Promise((resolve, reject) => {
		const ffmpeg = spawn("ffmpeg", [
			"-i",
			videoUrl,
			"-ss",
			"00:00:00.500",
			"-vframes",
			"1",
			"-f",
			"image2pipe",
			"-vcodec",
			"png",
			"-", // Output to stdout
		]);

		const chunks: Buffer[] = [];
		const errChunks: Buffer[] = [];

		ffmpeg.stdout.on("data", (chunk) => {
			chunks.push(chunk);
		});

		ffmpeg.stderr.on("data", (chunk) => {
			errChunks.push(chunk);
		});

		ffmpeg.on("close", async (code) => {
			if (code !== 0) {
				const errorMessage = Buffer.concat(errChunks).toString();
				mediaLogger.error({ err: errorMessage }, "FFmpeg error");
				return reject(
					new Error(
						`FFmpeg process exited with code ${code}. stderr: ${errorMessage.trim()}`,
					),
				);
			}

			const rawFrame = Buffer.concat(chunks);
			if (rawFrame.length === 0) {
				return reject(new Error("FFmpeg produced no output"));
			}

			try {
				const webpBuffer = await sharp(rawFrame)
					.resize({
						width,
						height,
						fit: "cover",
						position: "center",
					})
					.toFormat("webp", { quality: 80 })
					.toBuffer();
				resolve(webpBuffer);
			} catch (error) {
				reject(error);
			}
		});

		ffmpeg.on("error", (err) => {
			reject(err);
		});
	});
}

/**
 * Generates a thumbnail from an image buffer using Sharp.
 */
export async function generateImageThumbnail(
	imageBuffer: Buffer,
	width: number,
	height: number,
): Promise<Buffer> {
	return sharp(imageBuffer)
		.rotate() // Auto-rotate based on EXIF
		.resize({
			width,
			height,
			fit: "cover",
			position: "center",
		})
		.toFormat("webp", { quality: 70 })
		.toBuffer();
}

/**
 * Fetches a file from a URL and converts it to a base64 string
 */
export async function urlToBase64(url: string): Promise<string> {
	const response = await fetch(url);
	if (!response.ok)
		throw new Error(`Failed to fetch image: ${response.statusText}`);
	const arrayBuffer = await response.arrayBuffer();
	return Buffer.from(arrayBuffer).toString("base64");
}

/**
 * Normalize a WebM video to a constant 60fps using FFmpeg.
 *
 * Browser MediaRecorder output is variable-frame-rate and often has no
 * DefaultDuration, which makes container metadata report bogus values
 * (e.g. 1000fps). Re-encoding the video track with `fps=60` produces a
 * true, constant 60fps stream with accurate container metadata.
 */
export async function normalizeWebmTo60Fps(buffer: Buffer): Promise<Buffer> {
	const tempDir = os.tmpdir();
	const id = generateId();
	const tempIn = path.join(tempDir, `in_${id}.webm`);
	const tempOut = path.join(tempDir, `out_${id}.webm`);

	await fs.writeFile(tempIn, buffer);

	return new Promise((resolve, _reject) => {
		// Video: force constant 60fps. Audio: copy through untouched.
		// -vsync vfr + fps filter converts duplicate frames cleanly; -r 60
		// sets the container frame rate so metadata is honest.
		const ffmpeg = spawn("ffmpeg", [
			"-i",
			tempIn,
			"-map",
			"0:v:0",
			"-map",
			"0:a?",
			"-vf",
			"fps=60,format=yuv420p",
			"-r",
			"60",
			"-c:v",
			"libvpx-vp9",
			"-b:v",
			"8M",
			"-c:a",
			"copy",
			"-y",
			tempOut,
		]);

		let errorOutput = "";
		ffmpeg.stderr.on("data", (data) => {
			errorOutput += data.toString();
		});

		ffmpeg.on("close", async (code) => {
			try {
				if (code === 0) {
					const fixedBuffer = await fs.readFile(tempOut);
					resolve(fixedBuffer);
				} else {
					mediaLogger.warn(
						{ code, err: errorOutput },
						"ffmpeg normalizeWebmTo60fps failed",
					);
					resolve(buffer); // fallback to original
				}
			} catch (err) {
				mediaLogger.error({ err }, "Failed to read normalized webm");
				resolve(buffer);
			} finally {
				fs.unlink(tempIn).catch((err) => {
					mediaLogger.error(
						{ err, path: tempIn },
						"Cleanup: Failed to unlink tempIn",
					);
				});
				fs.unlink(tempOut).catch((err) => {
					mediaLogger.error(
						{ err, path: tempOut },
						"Cleanup: Failed to unlink tempOut",
					);
				});
			}
		});

		ffmpeg.on("error", (err) => {
			mediaLogger.error({ err }, "ffmpeg normalizeWebmTo60fps spawn error");
			resolve(buffer);
		});
	});
}

/**
 * Remuxes a WebM buffer using FFmpeg to fix missing track durations and metadata (kept for compat).
 */
export async function fixWebmMetadata(buffer: Buffer): Promise<Buffer> {
	const tempDir = os.tmpdir();
	const id = generateId();
	const tempIn = path.join(tempDir, `in_${id}.webm`);
	const tempOut = path.join(tempDir, `out_${id}.webm`);

	await fs.writeFile(tempIn, buffer);

	return new Promise((resolve, _reject) => {
		const ffmpeg = spawn("ffmpeg", ["-i", tempIn, "-c", "copy", "-y", tempOut]);

		let errorOutput = "";
		ffmpeg.stderr.on("data", (data) => {
			errorOutput += data.toString();
		});

		ffmpeg.on("close", async (code) => {
			try {
				if (code === 0) {
					const fixedBuffer = await fs.readFile(tempOut);
					resolve(fixedBuffer);
				} else {
					mediaLogger.warn(
						{ code, err: errorOutput },
						"ffmpeg fixWebmMetadata failed",
					);
					resolve(buffer); // fallback to original
				}
			} catch (err) {
				mediaLogger.error({ err }, "Failed to read fixed webm");
				resolve(buffer);
			} finally {
				fs.unlink(tempIn).catch((err) => {
					mediaLogger.error(
						{ err, path: tempIn },
						"Cleanup: Failed to unlink tempIn",
					);
				});
				fs.unlink(tempOut).catch((err) => {
					mediaLogger.error(
						{ err, path: tempOut },
						"Cleanup: Failed to unlink tempOut",
					);
				});
			}
		});

		ffmpeg.on("error", (err) => {
			mediaLogger.error({ err }, "ffmpeg fixWebmMetadata spawn error");
			resolve(buffer);
		});
	});
}

/**
 * Deeply calculates duration and FPS by running a full decode pass via FFmpeg.
 * Useful for browser WebM streams where headers are empty or inaccurate.
 */
export async function calculateFallbackMetadata(
	bufferOrUrl: Buffer | string,
): Promise<{ duration: number; fps: number } | null> {
	const isUrl = typeof bufferOrUrl === "string";
	let inputPath = "";
	let tempFile = "";

	if (isUrl) {
		inputPath = bufferOrUrl;
	} else {
		const tempDir = os.tmpdir();
		const id = generateId();
		tempFile = path.join(tempDir, `parse_${id}.webm`);
		await fs.writeFile(tempFile, bufferOrUrl);
		inputPath = tempFile;
	}

	return new Promise((resolve) => {
		const ffmpeg = spawn("ffmpeg", [
			"-v",
			"error",
			"-stats",
			"-i",
			inputPath,
			"-f",
			"null",
			"-",
		]);

		let output = "";
		ffmpeg.stderr.on("data", (data) => {
			// ffmpeg -stats outputs continuously to stderr
			output += data.toString();
		});

		ffmpeg.on("close", async () => {
			if (tempFile) {
				try {
					await fs.unlink(tempFile);
				} catch (err) {
					mediaLogger.error(
						{ err, path: tempFile },
						"Cleanup: Failed to unlink tempFile in fallback metadata",
					);
				}
			}

			// Look for the last "frame=  xxx" and "time=HH:MM:SS.xx"
			const frameMatch = [...output.matchAll(/frame=\s*(\d+)/g)].pop();
			const timeMatch = [
				...output.matchAll(/time=(\d{2}):(\d{2}):(\d{2}\.\d+)/g),
			].pop();

			if (timeMatch) {
				const hours = parseInt(timeMatch[1], 10);
				const minutes = parseInt(timeMatch[2], 10);
				const seconds = parseFloat(timeMatch[3]);
				const duration = hours * 3600 + minutes * 60 + seconds;

				let fps = 30; // default proxy
				if (frameMatch && duration > 0) {
					const frames = parseInt(frameMatch[1], 10);
					if (frames > 0) {
						fps = frames / duration;
					}
				}
				// ffmpeg's -stats frame counter can exceed real presented
				// frames for variable-frame-rate WebM (raw block counts),
				// producing absurd fps. Clamp to a sane playback range so
				// storage/UI never shows 1000fps for a real capture.
				if (Number.isFinite(fps) && (fps < 1 || fps > 120)) fps = 30;

				resolve({ duration, fps });
				return;
			}
			resolve(null);
		});

		ffmpeg.on("error", () => resolve(null));
	});
}

interface FFProbeAudioStream {
	codec_name?: string;
	sample_rate?: string;
	channels?: number;
	bits_per_sample?: number;
	bits_per_raw_sample?: string;
	bit_rate?: string;
}

interface FFProbeResult {
	streams?: FFProbeAudioStream[];
}

export async function getAudioDetails(bufferOrUrl: Buffer | string): Promise<{
	sampleRate: number | null;
	channels: number | null;
	bitDepth: number | null;
	audioCodec: string | null;
	audioBitrate: number | null;
} | null> {
	const { inputPath, cleanup } = await prepareProbeInput(
		bufferOrUrl,
		"temp_audio_details",
	);

	try {
		// 1. Check for audio stream and get parameters
		const { code, stdout, stderr } = await runFfprobe([
			"-v",
			"error",
			"-select_streams",
			"a:0",
			"-show_entries",
			"stream=codec_name,sample_rate,channels,bits_per_sample,bits_per_raw_sample,bit_rate",
			"-of",
			"json",
			inputPath,
		]);
		if (code !== 0) {
			throw new Error(`ffprobe exited with code ${code}. stderr: ${stderr}`);
		}
		let streamInfo: FFProbeAudioStream | null;
		try {
			const data = JSON.parse(stdout) as FFProbeResult;
			streamInfo = data.streams?.[0] || null;
		} catch (err) {
			throw new Error(`Failed to parse ffprobe output: ${err}`);
		}

		if (!streamInfo) {
			return null;
		}

		// Extract base properties
		const sampleRate = streamInfo.sample_rate
			? parseInt(streamInfo.sample_rate, 10)
			: null;
		const channels = streamInfo.channels != null ? streamInfo.channels : null;

		let bitDepth: number | null = null;
		if (streamInfo.bits_per_sample != null) {
			bitDepth = streamInfo.bits_per_sample;
		} else if (streamInfo.bits_per_raw_sample != null) {
			bitDepth = parseInt(streamInfo.bits_per_raw_sample, 10);
		}

		const audioCodec = streamInfo.codec_name || null;
		const audioBitrate = streamInfo.bit_rate
			? parseInt(streamInfo.bit_rate, 10)
			: null;

		return {
			sampleRate,
			channels,
			bitDepth,
			audioCodec,
			audioBitrate,
		};
	} catch (err) {
		mediaLogger.error({ err }, "Failed to extract audio details");
		return null;
	} finally {
		await cleanup("Failed to delete temp file in getAudioDetails");
	}
}
