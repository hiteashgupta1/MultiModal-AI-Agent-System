import requests
import cv2
import os
import uuid
import numpy as np
import time
import threading  
from backend.config import HF_TOKEN, VISION_MODEL
from backend.rag.memory_store import store_memory
from backend.evaluation.log_metrics import log_metric
from backend.experiments.experiment_tracker import log_experiment
from backend.config import TELEGRAM_TOKEN, CHAT_ID
from PIL import Image
from transformers import pipeline

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

SAVE_DIR = os.path.join(
    ROOT_DIR,
    "backend",
    "storage",
    "person_detections"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

ALERT_COOLDOWN = 30 
last_alert_time = 0

def send_telegram_or_system_alert(img_buffer, label):
    """Dispatches a notification and the photo payload to your Telegram channel."""
    try:
        # 1. Log the security breach into your RAG memory store
        store_memory(f"CRITICAL SECURITY ALERT: {label.capitalize()} detected.")

        # 2. Fire the photo over to Telegram via HTTP POST
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': ('alert.jpg', img_buffer, 'image/jpeg')}
        data = {
            'chat_id': CHAT_ID,
            'caption': f"⚠️ Alert: {label.capitalize()} detected by Vision Agent!"
        }
        
        response = requests.post(url, files=files, data=data, timeout=12)
        if response.status_code == 200:
            print("[ALERT DISPATCHER] Message successfully delivered to Telegram.")
        else:
            print(f"[ALERT ERROR] Telegram returned status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[ALERT CRITICAL FAILURE] Could not dispatch alert: {e}")

def detect_objects(image_bytes):
    global last_alert_time
    start_time = time.time()

    url = f"https://router.huggingface.co/hf-inference/models/{VISION_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
    }

    response = requests.post(url, headers=headers, data=image_bytes)

    if response.status_code != 200:
        print("VISION ERROR:", response.text)
        return None

    detections = response.json()

    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    detected_objects = []
    
    # Track if a person is found in the current frame/photo
    person_detected = False
    person_label = "person"

    for obj in detections:
        box = obj["box"]
        label = obj["label"]
        score = round(obj["score"], 2)
        print(f"[DEBUG] Model returned label: '{label}' | Score: {score}")
        x1 = int(box["xmin"])
        y1 = int(box["ymin"])
        x2 = int(box["xmax"])
        y2 = int(box["ymax"])
        label_clean = label.strip().lower()
        if "person" in label_clean:
            box_color = (0, 255, 0) # Green (BGR format)
            text_color = (0, 255, 0) # Green text
            person_detected = True
            person_label = label
        else:
            box_color = (0, 0, 255) # Red (BGR format)
            text_color = (0, 0, 255) # Red text
        # Always draw bounding boxes for EVERYTHING (cars, chairs, laptops, etc.)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        text = f"{label} ({score})"
        cv2.putText(
            img,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            2
        )

        detected_objects.append({
            "label": label,
            "confidence": score
        })

        # Check specifically for a person (case-insensitive)
        if label.lower() == "person":
            person_detected = True
            person_label = label
    # Process final annotated image buffer
    latency = round(time.time() - start_time, 2)
    _, buffer = cv2.imencode(".jpg", img)
    final_image_bytes = buffer.tobytes()

    # Always write to memory store for standard tracking logs
    if len(detected_objects) > 0:
        store_memory(f"Detected objects: {detected_objects}")
    saved_path = None
    # 🚨 CRITICAL: Conditional Alert Logic
    if person_detected:

        current_time = time.time()

        filename = f"{uuid.uuid4()}.jpg"

        saved_path = os.path.join(
            SAVE_DIR,
            filename
        )

        # SAVE IMAGE
        cv2.imwrite(
            saved_path,
            img
        )

        # =====================================
        # STORE IMAGE VECTOR MEMORY
        # =====================================

        from backend.rag.image_store import (
            store_person_image
        )

        store_person_image(saved_path)

        print(
            "✅ PERSON IMAGE STORED:",
            saved_path
        )
        
        if current_time - last_alert_time > ALERT_COOLDOWN:
            last_alert_time = current_time
            
            # Spin up alert in a background thread to prevent live streaming lag
            alert_thread = threading.Thread(
                target=send_telegram_or_system_alert, 
                args=(final_image_bytes, person_label)
            )
            alert_thread.start()

    # ✅ Calculate average confidence
    if len(detected_objects) > 0:
        avg_conf = round(
            sum(obj["confidence"] for obj in detected_objects) / len(detected_objects),
            2
        )
    else:
        avg_conf = 0.0

    # ✅ Log metrics (Keeps your experimental dashboard accurate for all data points)
    log_metric(
        agent="vision",
        model_version="objdetect_v1",
        confidence=avg_conf,
        object_count=len(detected_objects),
        latency=latency,
        hallucination=False
    )
    log_experiment(
        agent="vision",
        confidence=avg_conf,
        latency=latency,
        hallucination=False,
        model_version="objdetect_v1"
    )

    return {
        "image": final_image_bytes,
        "saved_path": saved_path,
        "person_detected": person_detected,
        "objects": detected_objects,
        "latency": latency,
        "confidence": avg_conf
    }