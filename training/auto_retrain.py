import json
import datetime
import pandas as pd
import os

MODEL_DIR = "models"
MODEL_PREFIX = "orchestrator_v"

FEEDBACK_FILE = "backend/experiments/feedback.csv"

def check_retrain():

    df = pd.read_csv(FEEDBACK_FILE)

    avg_rating = df["rating"].mean()

    if avg_rating < 3:

        print("Model quality dropped. Retraining...")

        os.system("python training/dataset_builder.py")

        os.system("python training/lora_train.py")

def get_next_model_version():

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    versions = []

    for name in os.listdir(MODEL_DIR):

        if name.startswith(MODEL_PREFIX):

            try:
                v = int(name.replace(MODEL_PREFIX, ""))
                versions.append(v)
            except:
                pass

    if not versions:
        return 1

    return max(versions) + 1