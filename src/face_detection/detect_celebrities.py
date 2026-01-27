import boto3
import os
import json
import logging
from PIL import Image

UNIQUE_DIR = "frames/unique"

# Configure logger for the module
from logging_utils import get_logger
logger = get_logger(__name__)



rek = boto3.client("rekognition")

def recognize_frame(image_path):
    with open(image_path, "rb") as img_file:
        bytes_data = img_file.read()
    
    logger.info(f"Calling Rekognition for image: {image_path}")
    response = rek.recognize_celebrities(Image={"Bytes": bytes_data})
    return response.get("CelebrityFaces", [])

    

def run_celebrity_detection():
    logger.info("Running AWS Rekognition Celebrity Detection...")

    if not os.path.exists(UNIQUE_DIR):
        logger.error("Unique faces directory not found.")
        return
    
    call_count = 0

    for filename in sorted(os.listdir(UNIQUE_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        if call_count >= 25:
            logger.warning("Reached Rekognition call limit (25). Stopping.")
            break

        img_path = os.path.join(UNIQUE_DIR, filename)
        logger.info(f"Processing: {img_path}")

        celebs = recognize_frame(img_path)
        call_count += 1

        if not celebs:
            logger.info("No celebrity recognized.")
            continue

        # Log results
        for celeb in celebs:
            name = celeb.get("Name")
            match_confidence = celeb.get("MatchConfidence", 0.0)
            urls = celeb.get("Urls", [])

            logger.info(f"Celebrity Detected: {name}")
            logger.info(f"Confidence: {match_confidence:.2f}")
            logger.info(f"URLs: {urls}")

if __name__ == "__main__":
    run_celebrity_detection()