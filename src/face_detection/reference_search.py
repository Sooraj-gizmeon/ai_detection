import json
import numpy as np
from numpy.linalg import norm
from .object_hash import get_embedding
import cv2
from .object_detect import detect_objects
import os


import logging
logger = logging.getLogger('integrated_processor')

from .utils import enrich_result_with_reference_scores

# ------------------ Similarity ------------------
def cosine_sim(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b)))


def frame_to_sec(frame, fps):
    return frame / fps


# ------------------ Reference Image Search ------------------
def search_reference_image(
    reference_image_path,
    video_bucket,
    result_json_path="output/result.json",
    index_path="frames/object_index.json",
    similarity_threshold=0.75,
    fps=1
):
    """
    Find which existing object segments contain the reference image.
    """
    logger.info("Starting reference image search...")

    # Load object index (unique object embeddings)
    logger.info("Loading object index file")
    with open(index_path) as f:
        index = json.load(f)

    # Load pipeline results (segments)
    logger.info("Loading pipeline result")
    with open(result_json_path) as f:
        results = json.load(f)

    logger.info("Similarity threshold: %.2f", similarity_threshold)
    
    # Detect object in reference image and compute embedding for the object only
    logger.info("Detecting object in reference image before embedding...")

    input_dir = "/app/input"
    os.makedirs(input_dir, exist_ok=True)

    local_image_path = os.path.join(
        input_dir, os.path.basename(reference_image_path)
    )
    
    # Download reference image if it's a URL or if local file doesn't exist
    if not os.path.exists(local_image_path):
        if reference_image_path.startswith("http://") or reference_image_path.startswith("https://"):
            logger.info(f"Downloading reference image from {reference_image_path} to {local_image_path}")
            import urllib.request
            urllib.request.urlretrieve(reference_image_path, local_image_path)
        else:
            # Assume it's a local path, copy it
            import shutil
            shutil.copy2(reference_image_path, local_image_path)
            logger.info(f"Copied reference image from {reference_image_path} to {local_image_path}")
    
    img = cv2.imread(local_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {local_image_path}")

    objects = detect_objects(img)
    if not objects:
        raise ValueError(f"No object detected in reference image: {local_image_path}")
    # Use the first detected object (or implement logic to select the best one)
    x1, y1, x2, y2 = objects[0]["bbox"]
    obj_crop = img[y1:y2, x1:x2]
    logger.info(f"Object detected and cropped: bbox=({x1}, {y1}, {x2}, {y2})")
    ref_emb = get_embedding(obj_crop)

    matched_objects = []

    # Compare reference to each unique object in index
    for item in index:
        emb = np.array(item["embedding"])
        score = cosine_sim(ref_emb, emb)
        logger.info(f"Comparing to unique object {item['object_id']} (first at frame {item['frame']}): score={score:.4f}")

        if score >= similarity_threshold:
            matched_objects.append({
                "object_id": item["object_id"],
                "label": item["label"],
                "score": round(score, 3),
                "first_frame": item["frame"]
            })

    # Collect all segments from matched objects
    matched_segments = []
    added_segments = set()

    for obj in results.get("objects", []):
        logger.info(f"Checking object {obj['object_id']} in results for matches...")
        for matched_obj in matched_objects:
            if obj["object_id"] == f"object_{matched_obj['object_id']}":
                for seg in obj.get("segments", []):
                    segment_key = (obj["object_id"], seg["start_sec"], seg["end_sec"])
                    if segment_key not in added_segments:
                        logger.info(f"Found matching segment for object {obj['object_id']}: {seg['start_time']} - {seg['end_time']} (score={matched_obj['score']})")
                        matched_segments.append({
                            "object_id": obj["object_id"],
                            "label": obj["label"],
                            "start_sec": seg["start_sec"],
                            "end_sec": seg["end_sec"],
                            "start_time": seg["start_time"],
                            "end_time": seg["end_time"],
                            "score": matched_obj["score"]
                        })
                        added_segments.add(segment_key)
                        logger.info(f"Added segment with score: {segment_key}")
                break  # No need to check other matched_objects for this obj
                    
    logger.info(f"Found {len(matched_segments)} matching segments for reference image.")
    logger.info("Reference image search completed.")
    
    # Enrich the result.json with reference scores
    enrich_result_with_reference_scores(
        result_json_path=result_json_path,
        matched_segments=matched_segments,
    )
    
    return {
        "reference_image": reference_image_path,
        "total_matches": len(matched_segments),
        "segments": matched_segments
    }