import os
import glob
import json

from face_detection.extract_frames import extract_frames
from .face_hash import process_unique_faces
from .detect_celebrities import recognize_frame
from .object_hash import process_unique_objects
from .utils import (
    ensure_dir,
    save_json,
    timestamp_from_frame,
    timestamps_to_segments,
    seconds_to_hhmmss,
)

from logging_utils import get_logger

logger = get_logger(__name__)


def run_pipeline(
    video_path,
    fps=1,
    output_json="output/result.json",
    raw_dir="/app/frames/raw",
    unique_faces_dir="/app/frames/unique",
    unique_objects_dir="/app/frames/unique_objects",
    celebrity_detection=True,
    object_detection=True,
):
    # -----------------------------
    # Setup
    # -----------------------------
    ensure_dir(raw_dir)
    ensure_dir(unique_faces_dir)
    ensure_dir(unique_objects_dir)

    logger.info(f"Pipeline started | video={video_path} | fps={fps}")

    # -----------------------------
    # 1. Extract Frames
    # -----------------------------
    logger.info("[1] Extracting frames...")
    extract_frames(video_path, raw_dir, fps=fps)

    # -----------------------------
    # 2. FACE PIPELINE
    # -----------------------------
    celebrity_results = []
    if celebrity_detection:
        logger.info("[2] Processing unique faces...")
        process_unique_faces(fps=fps)

        face_appearance_map_path = os.path.join(
            unique_faces_dir, "face_appearance_map.json"
        )

        face_appearance_map = (
            json.load(open(face_appearance_map_path))
            if os.path.exists(face_appearance_map_path)
            else {}
        )

        face_images = sorted(
            glob.glob(f"{unique_faces_dir}/*.jpg")
        )

        # Limit to 30 unique faces for celebrity recognition
        max_faces = 30
        face_images = face_images[:max_faces]

        logger.info("[3] Running celebrity recognition (AWS Rekognition)...")
        logger.info("Total unique faces to process: %d (limited to %d)", len(face_images), max_faces)

        for face_img in face_images:
            face_file = os.path.basename(face_img)
            appearances = face_appearance_map.get(face_file, [])

            celebs = recognize_frame(face_img)
            logger.info("Celebrities recognized for %s: %d", face_file, len(celebs))
            if not celebs:
                continue

            for celeb in celebs:
                name = celeb.get("Name")
                conf = round(
                    celeb.get("MatchConfidence", 0.0) / 100.0, 2
                )

                timestamps = []
                for frame_num in appearances:
                    sec, ts = timestamp_from_frame(int(frame_num), fps)
                    timestamps.append(
                        {
                            "timestamp_sec": int(sec),
                            "timestamp": ts,
                        }
                    )

                celebrity_results.append(
                    {
                        "name": name,
                        "face_id": face_file,
                        "confidence": conf,
                        "appearances": timestamps,
                    }
                )
        logger.info("Celebrity results: %s", celebrity_results)

        logger.info(f"✅ Recognized {len(celebrity_results)} celebrity appearances")
    else:
        logger.info("Skipping celebrity detection")

    # -----------------------------
    # 3. OBJECT PIPELINE
    # -----------------------------
    object_results = []
    if object_detection:
        logger.info("[4] Processing unique objects...")

        # IMPORTANT: process_unique_objects RETURNS A SINGLE LIST
        object_records = process_unique_objects(
            target_labels=["bottle", "cup"]
        )

        # Group frames by object_id
        object_frame_map = {}

        for rec in object_records:
            obj_id = rec["object_id"]
            label = rec["label"]
            frame = rec["frame"]

            if obj_id not in object_frame_map:
                object_frame_map[obj_id] = {
                    "label": label,
                    "frames": []
                }

            object_frame_map[obj_id]["frames"].append(frame)

        for obj_id, data in object_frame_map.items():
            frames = data["frames"]

            # Convert frames → seconds
            timestamps_sec = sorted(
                {
                    int(timestamp_from_frame(int(f), fps)[0])
                    for f in frames
                }
            )

            # Convert seconds → segments
            raw_segments = timestamps_to_segments(
                timestamps_sec, gap=2
            )

            segments = [
                {
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "start_time": seconds_to_hhmmss(seg["start_sec"]),
                    "end_time": seconds_to_hhmmss(seg["end_sec"]),
                }
                for seg in raw_segments
            ]

            object_results.append(
                {
                    "object_id": f"object_{obj_id}",
                    "label": data["label"],
                    "segments": segments,
                }
            )
    else:
        logger.info("Skipping object detection")

    # -----------------------------
    # 4. Unified Output
    # -----------------------------
    final_output = {
        "video": os.path.basename(video_path),
        "fps": fps,
        "celebrities": celebrity_results,
        "objects": object_results,
    }

    # -----------------------------
    # 5. Save Results
    # -----------------------------
    logger.info("[5] Saving results...")
    ensure_dir(os.path.dirname(output_json))
    save_json(output_json, final_output)

    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    return final_output
