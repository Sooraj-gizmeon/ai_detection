# object_detect.py
from ultralytics import YOLO
import logging
from logging_utils import get_logger

logger = get_logger(__name__)

# object_model = YOLO("models/yolov8x.pt")
#object_model = YOLO("models/yolov8n_100e.pt")
object_model = YOLO("yolov8m.pt")

def detect_objects(img, class_names=None, conf_thresh=0.4):
    logger.info("Detecting objects in image...")
    results = object_model(img, device=0)[0]
    objects = []

    for box in results.boxes:
        cls_id = int(box.cls)
        label = object_model.names[cls_id]
        conf = float(box.conf)

        if conf < conf_thresh:
            continue

        if class_names and label not in class_names:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        objects.append({
            "label": label,
            "bbox": (x1, y1, x2, y2),
            "confidence": conf
        })
    logger.info(f"Detected {len(objects)} objects.")
    return objects



if __name__ == "__main__":
    logger.info("Running YOLO + CLIP Unique Object Extraction...")
    detect_objects()