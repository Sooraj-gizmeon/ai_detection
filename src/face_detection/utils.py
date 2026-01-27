import os
import json
from datetime import timedelta
import subprocess
import logging
import sys
from functools import lru_cache


from logging_utils import get_logger
logger = get_logger(__name__)


def timestamp_from_frame(frame_number, fps):
    sec = (frame_number - 1) / float(fps)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return sec, f"{h:02d}:{m:02d}:{s:02d}"



def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)



def get_video_fps(video_path):
    """
    Returns the TRUE FPS of the input video using ffprobe.
    Example output: 29.97, 24.0, 30.0, 23.976, etc.
    """
    cmd = [
        "ffprobe", "-v", "0", "-of", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    info = json.loads(result.stdout)
    frame_rate = info["streams"][0]["r_frame_rate"]  # e.g., "30000/1001"

    num, den = frame_rate.split("/")
    return float(num) / float(den)    




def get_video_codec(input_video):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_video
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()




def frames_to_segments(matches, fps=1, gap=2):
    segments = []
    current = [matches[0]]

    for m in matches[1:]:
        if m["frame"] - current[-1]["frame"] <= gap:
            current.append(m)
        else:
            segments.append(current)
            current = [m]

    segments.append(current)

    return [
        {
            "start_sec": seg[0]["frame"] / fps,
            "end_sec": seg[-1]["frame"] / fps
        }
        for seg in segments
    ]



# -----------------------------
# Helper: timestamps → segments
# -----------------------------
def timestamps_to_segments(timestamps, gap=2):
    """
    timestamps: sorted list of seconds
    gap: max allowed gap (seconds) to consider continuous segment
    """
    if not timestamps:
        return []

    segments = []
    start = prev = timestamps[0]

    for t in timestamps[1:]:
        if t - prev <= gap:
            prev = t
        else:
            segments.append(
                {
                    "start_sec": start,
                    "end_sec": prev,
                }
            )
            start = prev = t

    segments.append(
        {
            "start_sec": start,
            "end_sec": prev,
        }
    )

    return segments


def seconds_to_hhmmss(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



# def get_logger(name: str):
#     return logging.getLogger(name)




def enrich_result_with_reference_scores(
    result_json_path,
    matched_segments,
):
    """
    Inject reference image similarity scores into existing result.json
    """

    logger.info("Enriching result.json with reference scores...")

    with open(result_json_path) as f:
        result = json.load(f)

    for obj in result.get("objects", []):
        for seg in obj.get("segments", []):
            for match in matched_segments:
                if (
                    obj["object_id"] == match["object_id"]
                    and seg["start_sec"] == match["start_sec"]
                    and seg["end_sec"] == match["end_sec"]
                ):
                    seg["reference_match"] = {
                        "score": match["score"]
                    }

    with open(result_json_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Result enrichment completed.")