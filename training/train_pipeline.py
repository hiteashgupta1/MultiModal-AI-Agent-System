import subprocess
import pandas as pd
import os

FEEDBACK_FILE = "backend/experiments/feedback.csv"
RETRAIN_THRESHOLD = 10   # retrain after 10 feedback samples


def should_retrain():

    if not os.path.exists(FEEDBACK_FILE):
        return False

    df = pd.read_csv(FEEDBACK_FILE, on_bad_lines="skip")

    if len(df) >= RETRAIN_THRESHOLD:
        return True

    return False


def run_pipeline():

    if not should_retrain():
        print("Retrain threshold not reached")
        return

    print("Retraining triggered")

    print("STEP 1: Building dataset")

    subprocess.run(
        ["python", "training/smart_dataset_builder.py"],
        check=True
    )

    print("STEP 2: Training model")

    subprocess.run(
        ["python", "training/auto_retrain.py"],
        check=True
    )

    print("Training pipeline completed")


if __name__ == "__main__":
    run_pipeline()