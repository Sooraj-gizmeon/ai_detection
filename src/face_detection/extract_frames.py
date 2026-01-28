import os
import subprocess
import argparse

from .utils import get_video_codec, get_video_fps
import logging
logger = logging.getLogger('integrated_processor')


def extract_frames(input_video, output_dir, fps=1, use_gpu=True):
    logger.info("=== FRAME EXTRACTION STARTED FROM Src.Extract_frames ===")
    logger.info(f"Input video path: {input_video}")

    if not os.path.exists(input_video):
        logger.error(f"Video not found: {input_video}")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Detect real FPS of input video
    real_fps = get_video_fps(input_video)
    logger.info(f"Real Video FPS: {real_fps}")

    codec = get_video_codec(input_video)
    logger.info(f"Detected video codec: {codec}")

    # Map input codec to GPU decoder
    gpu_decoder_map = {
        "h264": "h264_cuvid",
        "hevc": "hevc_cuvid",
        "av1": "av1_cuvid",
        "mpeg4": "mpeg4_cuvid",
        "vp8": "vp8_cuvid",
        "vp9": "vp9_cuvid",
        
    }

    decoder = gpu_decoder_map.get(codec, None)
    logger.info(f"Using decoder: {decoder}")

    # Primary ffmpeg command (GPU if requested and supported)
    if use_gpu and decoder:
        logger.info("Using GPU-accelerated FFmpeg pipeline.")
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-c:v",
            decoder,  # ensure the CUDA decoder is actually used
            "-i",
            input_video,
            "-vf",
            f"fps={fps},hwdownload,format=nv12",
            os.path.join(output_dir, "frame_%04d.jpg"),
            "-y",
        ]
    else:
        if use_gpu and not decoder:
            logger.warning(
                "GPU requested but no matching hardware decoder for codec '%s'. "
                "Falling back to CPU pipeline.",
                codec,
            )
        logger.info("Using CPU FFmpeg pipeline.")
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_video,
            "-vf",
            f"fps={fps}",
            os.path.join(output_dir, "frame_%04d.jpg"),
            "-y",
        ]

    logger.info("Running FFmpeg command:")
    logger.info(" ".join(ffmpeg_cmd))

    # Run primary FFmpeg command
    process = subprocess.run(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logger.info("===== FFmpeg Output =====")
    if process.stdout:
        logger.info(process.stdout)
    if process.stderr:
        logger.info(process.stderr)
    logger.info("=========================")

    # If GPU pipeline failed, fall back to CPU
    if process.returncode != 0 and use_gpu:
        logger.warning(
            "FFmpeg command failed with return code %s. "
            "Falling back to CPU pipeline.",
            process.returncode,
        )
        cpu_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_video,
            "-vf",
            f"fps={fps}",
            os.path.join(output_dir, "frame_%04d.jpg"),
            "-y",
        ]
        logger.info("Running CPU fallback FFmpeg command:")
        logger.info(" ".join(cpu_cmd))

        process = subprocess.run(
            cpu_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        logger.info("===== FFmpeg CPU Fallback Output =====")
        if process.stdout:
            logger.info(process.stdout)
        if process.stderr:
            logger.info(process.stderr)
        logger.info("======================================")

    # Count frames
    files = os.listdir(output_dir)
    total = len([f for f in files if f.endswith(".jpg")])
    logger.info(f"Total frames extracted: {total}")
    logger.info("=== EXTRACTION COMPLETE ===")
    return real_fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    args = parser.parse_args()

    extract_frames(
        args.input,
        "frames/raw",
        fps=args.fps,
        use_gpu=not args.cpu,
    )