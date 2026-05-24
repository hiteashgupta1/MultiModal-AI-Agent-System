import csv
import os
from datetime import datetime

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
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

FEEDBACK_FILE = os.path.join(
    EVAL_DIR,
    "feedback.csv"
)

def save_feedback(input_text, rating, comment):


    file_exists = os.path.exists(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only if file does not exist
        if not file_exists:
            writer.writerow(["timestamp", "input_text", "rating", "comment"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_text,
            rating,
            comment
        ])