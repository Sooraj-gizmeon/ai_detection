# src/celebrity/celebrity_index.py
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_celebrity_index(path: str) -> Tuple[Dict[str, List[int]], Dict[str, float]]:
    """
    Load result.json and build:
      - appearances_per_actor: actor -> sorted list of integer seconds where they appear
      - actor_conf: actor -> global confidence
    
    IMPORTANT: Same actor can be detected multiple times with different face_ids.
    This function consolidates all appearances for the same actor name.
    """
    data = json.loads(Path(path).read_text())
    appearances_per_actor: Dict[str, List[int]] = {}
    actor_conf: Dict[str, float] = {}

    for celeb in data.get("celebrities", []):
        name = celeb.get("name")
        if not name:
            continue
        conf = float(celeb.get("confidence", 1.0))
        times = sorted(int(a["timestamp_sec"]) for a in celeb.get("appearances", []) if "timestamp_sec" in a)
        if not times:
            continue
        
        # Consolidate appearances for same actor (same actor detected with different face_ids)
        if name not in appearances_per_actor:
            appearances_per_actor[name] = []
            actor_conf[name] = conf
        
        appearances_per_actor[name].extend(times)
    
    # Remove duplicates and sort for each actor
    for actor in appearances_per_actor:
        appearances_per_actor[actor] = sorted(set(appearances_per_actor[actor]))

    return appearances_per_actor, actor_conf


def load_object_index(path: str) -> Tuple[Dict[str, List[Dict]], Dict[str, float], Dict[str, str]]:
    """
    Load result.json for object detection data.
    Returns:
      - appearances_per_object: object_id -> list of dicts with 'start_sec', 'end_sec', 'score'
      - object_conf: object_id -> confidence score
      - object_labels: object_id -> label/name
    
    This function handles object detection results similar to celebrity detection.
    """
    data = json.loads(Path(path).read_text())
    appearances_per_object: Dict[str, List[Dict]] = {}
    object_conf: Dict[str, float] = {}
    object_labels: Dict[str, str] = {}

    # If objects exist in the data, process them
    for obj in data.get("objects", []):
        obj_id = obj.get("id", obj.get("label", "unknown"))
        label = obj.get("label", obj_id)
        conf = float(obj.get("confidence", 1.0))
        
        if obj_id not in appearances_per_object:
            appearances_per_object[obj_id] = []
            object_conf[obj_id] = conf
            object_labels[obj_id] = label
        
        # Process segments/appearances for this object
        for segment in obj.get("segments", obj.get("appearances", [])):
            appearances_per_object[obj_id].append({
                "start_sec": float(segment.get("start_sec", 0)),
                "end_sec": float(segment.get("end_sec", 0)),
                "score": float(segment.get("score", conf)),
            })

    return appearances_per_object, object_conf, object_labels


def actor_coverage_for_segment(
    start_sec: float,
    end_sec: float,
    appearances_per_actor: Dict[str, List[int]],
    actor_conf: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    For a segment [start_sec, end_sec], compute per-actor:
      coverage: fraction of segment seconds containing that actor
      confidence: actor-level confidence from result.json
    """
    seg_len = max(end_sec - start_sec, 1e-6)
    per_actor: Dict[str, Dict[str, float]] = {}

    for actor, times in appearances_per_actor.items():
        # timestamps are at 1s resolution, treat them as “seconds present”
        count = sum(1 for t in times if start_sec <= t <= end_sec)
        if count == 0:
            continue
        coverage = count / seg_len
        per_actor[actor] = {
            "coverage": min(1.0, coverage),
            "confidence": float(actor_conf.get(actor, 1.0)),
        }

    return per_actor


def compute_celebrity_score(
    per_actor: Dict[str, Dict[str, float]],
    important_actors: Dict[str, float] = None,
) -> float:
    """
    Aggregate per-actor coverage into a single celebrity_score for the segment.
    important_actors: optional explicit weights, e.g. leads.
    """
    if not per_actor:
        return 0.0

    important_actors = important_actors or {}
    score = 0.0
    for actor, feat in per_actor.items():
        w = important_actors.get(actor, 1.0)
        score += w * feat["coverage"] * feat["confidence"]

    # Normalize mildly to keep it in a sane range (0–1-ish)
    return min(1.0, score)


def object_coverage_for_segment(
    start_sec: float,
    end_sec: float,
    appearances_per_object: Dict[str, List[Dict]],
    object_conf: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    For a segment [start_sec, end_sec], compute per-object:
      coverage: fraction of segment seconds containing that object
      score: object-level score from reference_match
    """
    seg_len = max(end_sec - start_sec, 1e-6)
    per_object: Dict[str, Dict[str, float]] = {}

    for object_id, segments in appearances_per_object.items():
        # Check if any object segment overlaps with the segment
        overlap_duration = 0.0
        max_score = 0.0
        for obj_seg in segments:
            obj_start = obj_seg['start_sec']
            obj_end = obj_seg['end_sec']
            score = obj_seg['score']
            
            # Calculate overlap
            overlap_start = max(start_sec, obj_start)
            overlap_end = min(end_sec, obj_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                overlap_duration += overlap
                max_score = max(max_score, score)
        
        if overlap_duration > 0:
            coverage = overlap_duration / seg_len
            per_object[object_id] = {
                "coverage": min(1.0, coverage),
                "score": float(max_score),
            }

    return per_object


def compute_object_score(
    per_object: Dict[str, Dict[str, float]],
    important_objects: Dict[str, float] = None,
) -> float:
    """
    Aggregate per-object coverage into a single object_score for the segment.
    important_objects: optional explicit weights.
    """
    if not per_object:
        return 0.0

    important_objects = important_objects or {}
    score = 0.0
    for object_id, feat in per_object.items():
        w = important_objects.get(object_id, 1.0)
        score += w * feat["coverage"] * feat["score"]

    # Normalize mildly to keep it in a sane range (0–1-ish)
    return min(1.0, score)
