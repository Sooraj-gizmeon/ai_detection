# object_hash.py
import os
import cv2
import json
import torch
import clip
import numpy as np
from PIL import Image
from numpy.linalg import norm
from .object_detect import detect_objects

from logging_utils import get_logger

logger = get_logger(__name__)

# -------------------- Paths --------------------
RAW_DIR = "frames/raw"
UNIQUE_OBJ_DIR = "frames/unique_objects"
INDEX_PATH = "frames/object_index.json"

os.makedirs(UNIQUE_OBJ_DIR, exist_ok=True)

# -------------------- Device & Model --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ⚠️ Recommended for production (faster, stable)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# -------------------- Utils --------------------
def get_embedding(img):
    """Generate CLIP embedding for an image crop."""
    logger.info("Generating CLIP embedding for object...")
    img_pil = Image.fromarray(img)
    image_input = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = clip_model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy().astype("float32")[0]


def cosine_similarity(a, b):
    logger.info("Calculating cosine similarity...")
    return float(np.dot(a, b) / (norm(a) * norm(b)))


# -------------------- Main Indexing Function --------------------
def process_unique_objects(
    target_labels=None,
    similarity_threshold=0.75,
):
    """
    Extract objects from frames, compute CLIP embeddings,
    deduplicate, and persist an object index for search.
    """

    logger.info("🚀 Starting object indexing pipeline")

    records = []
    seen_embeddings = []

    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.endswith(".jpg"):
            continue

        img_path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        frame_num = int(filename.replace("frame_", "").replace(".jpg", ""))

        # YOLO detection
        objects = detect_objects(img, class_names=target_labels)

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            emb = get_embedding(crop)

            matched = False
            for prev in seen_embeddings:
                if cosine_similarity(emb, prev) >= similarity_threshold:
                    matched = True
                    break

            if matched:
                continue

            # Save unique object crop
            obj_id = len(seen_embeddings)
            obj_filename = f"object_{obj_id:04d}.jpg"
            Image.fromarray(crop).save(os.path.join(UNIQUE_OBJ_DIR, obj_filename))

            # Store record
            records.append({
                "object_id": obj_id,
                "label": obj["label"],
                "frame": frame_num,
                "image": obj_filename,
                "embedding": emb.tolist()
            })

            seen_embeddings.append(emb)

    # Persist index
    with open(INDEX_PATH, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"✅ Indexed {len(records)} unique objects")
    logger.info(f"📦 Object index saved to {INDEX_PATH}")

    return records


# -------------------- CLI --------------------
if __name__ == "__main__":
    logger.info("Running YOLO + CLIP Object Indexing")
    process_unique_objects()