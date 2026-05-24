import csv
import os
from datetime import datetime

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

# =========================================
# EVALUATION DIRECTORY
# =========================================

EVAL_DIR = os.path.join(
    ROOT_DIR,
    "backend",
    "evaluation"
)

os.makedirs(
    EVAL_DIR,
    exist_ok=True
)

METRICS_FILE = os.path.join(
    EVAL_DIR,
    "metrics.csv"
)

def log_metric(agent, model_version, confidence, latency, hallucination, object_count=None):


    file_exists = os.path.exists(METRICS_FILE)

    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "agent",
                "model_version",
                "confidence",
                "latency",
                "hallucination",
                "object_count"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            agent,
            model_version,
            confidence,
            latency,
            hallucination,
            object_count
        ])