import pandas as pd
import json
import os

METRICS_FILE = "backend/experiments/metrics.csv"
FEEDBACK_FILE = "backend/experiments/feedback.csv"
EXPERIMENT_FILE = "backend/backend\experiments/experiments.json"

OUTPUT_FILE = "training/train.json"


def load_metrics():

    if not os.path.exists(METRICS_FILE):
        return pd.DataFrame()

    return pd.read_csv(
        METRICS_FILE,
        on_bad_lines="skip",
        engine="python"
    )


def load_feedback():

    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()

    return pd.read_csv(
        FEEDBACK_FILE,
        on_bad_lines="skip",
        engine="python"
    )


def load_experiments():

    if not os.path.exists(EXPERIMENT_FILE):
        return []

    with open(EXPERIMENT_FILE) as f:
        return json.load(f)


def build_dataset():

    dataset = []

    metrics = load_metrics()
    feedback = load_feedback()
    experiments = load_experiments()

    print("Loaded metrics:", len(metrics))
    print("Loaded feedback:", len(feedback))
    print("Loaded experiments:", len(experiments))

    # -------------------------
    # 1️⃣ High confidence outputs
    # -------------------------

    if not metrics.empty:

        for _, row in metrics.iterrows():

            if "confidence" not in row:
                continue

            confidence = float(row.get("confidence", 0))

            if confidence < 0.7:
                continue

            if row.get("hallucination", False):
                continue

            input_text = str(row.get("input_text", ""))
            output = str(row.get("summary", ""))

            if len(input_text) < 10:
                continue

            dataset.append({
                "input": input_text,
                "output": output
            })

    # -------------------------
    # 2️⃣ Good user feedback
    # -------------------------

    if not feedback.empty:

        for _, row in feedback.iterrows():

            rating = int(row.get("rating", 0))

            if rating < 4:
                continue

            text = str(row.get("input_text", ""))

            if len(text) < 10:
                continue

            dataset.append({
                "input": text,
                "output": text
            })

    # -------------------------
    # 3️⃣ Experiment results
    # -------------------------

    for exp in experiments:

        confidence = exp.get("confidence", 0)

        if confidence < 0.75:
            continue

        dataset.append({
            "input": exp.get("input_text", ""),
            "output": exp.get("summary", "")
        })

    # Remove duplicates
    unique = []
    seen = set()

    for item in dataset:

        key = item["input"]

        if key not in seen:
            unique.append(item)
            seen.add(key)

    dataset = unique

    os.makedirs("training", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print("Final dataset size:", len(dataset))


if __name__ == "__main__":
    build_dataset()