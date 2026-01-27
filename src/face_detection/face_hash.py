import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import logging
from logging_utils import get_logger



RAW_DIR = "frames/raw"
UNIQUE_DIR = "frames/unique"
os.makedirs(UNIQUE_DIR, exist_ok=True)

# Configure logger for the module
logger = get_logger(__name__)

model = YOLO("models/yolov8n_100e.pt")
face_app = FaceAnalysis(name="buffalo_l")
face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# face_app = FaceAnalysis(
#     name="antelopev2",
#     root="/root/.insightface/models", 
#     providers=["CUDAExecutionProvider"]
# )
# face_app.prepare(ctx_id=0, det_size=(640, 640))


def get_embedding(face_img_rgb):
    logger.info("Generating InsightFace embedding for face...")
    face = face_app.get(face_img_rgb)
    if len(face) == 0:
        return None
    return face[0].embedding.astype('float32')

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def detect_faces_yolo(img):
    results = model(img, device=0)[0]
    faces = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf)
        logger.info(f"YOLO detected box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={conf:.3f}")
        if conf < 0.58:
            logger.info(f"Skipping box due to low confidence: {conf:.3f}")
            continue
        w, h = x2 - x1, y2 - y1
        if w < 80 or h < 80:
            logger.info(f"Skipping box due to small size: w={w}, h={h}")
            continue
        pad = int(0.25 * h)
        x1 = max(0, int(x1 - pad))
        y1 = max(0, int(y1 - pad))
        x2 = int(x2 + pad)
        y2 = int(y2 + pad)
        faces.append((x1, y1, x2, y2))
        logger.info(f"Accepted face: ({x1}, {y1}, {x2}, {y2})")
    logger.info(f"Total faces detected in image: {len(faces)}")
    return faces

def process_unique_faces(fps=1):
    import json
    seen_embeddings = []
    appearances = []
    threshold = 0.45
    face_to_frame_map = {}
    all_face_appearances = {}

    for filename in sorted(os.listdir(RAW_DIR)):
        img_path = os.path.join(RAW_DIR, filename)
        img = cv2.imread(img_path)
        frame_num = int(filename.replace("frame_", "").replace(".jpg", ""))
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
        logger.info(f"Processing frame: {filename}")
        faces = detect_faces_yolo(img)
        logger.info(f"Number of faces detected in {filename}: {len(faces)}")
        for (x1, y1, x2, y2) in faces:
            logger.info(f"Processing detected face crop: ({x1}, {y1}, {x2}, {y2}) in {filename}")
            face = img[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            emb = get_embedding(face_rgb)
            if emb is None:
                logger.info(f"No embedding found for face in {filename}")
                continue
            match_idx = None
            for i, prev_emb in enumerate(seen_embeddings):
                sim = cosine_similarity(emb, prev_emb)
                logger.info(f"Similarity with unique face {i}: {sim:.3f}")
                if sim > threshold:
                    match_idx = i
                    logger.info(f"Face in {filename} matched with unique face {i} (sim={sim:.3f})")
                    break
            if match_idx is not None:
                # Existing face, add appearance
                appearances[match_idx].append(frame_num)
                logger.info(f"Added appearance for existing unique face {match_idx} in frame {frame_num}")
            else:
                # Unique face
                seen_embeddings.append(emb)
                appearances.append([frame_num])
                face_idx = len(seen_embeddings)
                face_filename = f"face_{face_idx:04d}.jpg"
                save_path = os.path.join(UNIQUE_DIR, face_filename)
                Image.fromarray(face_rgb).save(save_path)
                face_to_frame_map[face_filename] = filename
                logger.info(f"Saved new unique face {face_filename} from frame {filename}")

    # Save all face appearance mapping
    all_face_appearances = {
        f"face_{i+1:04d}.jpg": frames
        for i, frames in enumerate(appearances)
    }

    json.dump(face_to_frame_map, open(os.path.join(UNIQUE_DIR, "face_to_frame_map.json"), "w"), indent=2)
    json.dump(all_face_appearances, open(os.path.join(UNIQUE_DIR, "face_appearance_map.json"), "w"), indent=2)
    logger.info("Saved face-to-frame and appearance mapping.")

if __name__ == "__main__":
    logger.info("Running YOLO + InsightFace Unique Face Extraction...")
    process_unique_faces()