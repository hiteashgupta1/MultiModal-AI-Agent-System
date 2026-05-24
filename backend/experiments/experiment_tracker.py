import json
import os

from datetime import datetime

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


EXP_DIR = os.path.join(
    ROOT_DIR,
    "backend",
    "experiments"
)

os.makedirs(
    EXP_DIR,
    exist_ok=True
)

EXP_FILE = os.path.join(
    EXP_DIR,
    "experiments.json"
)

def log_experiment(
    agent,
    confidence,
    latency,
    hallucination,
    model_version
):

    # =====================================
    # EXPERIMENT OBJECT
    # =====================================

    experiment = {

        "timestamp": str(
            datetime.now()
        ),

        "agent": str(agent),

        "confidence": float(confidence),

        "latency": float(latency),

        "hallucination": bool(hallucination),

        "model_version": str(model_version)
    }

    # =====================================
    # LOAD EXISTING DATA
    # =====================================

    if os.path.exists(EXP_FILE):

        try:

            with open(
                EXP_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

        except:

            data = []

    else:

        data = []

    # =====================================
    # APPEND NEW ENTRY
    # =====================================

    data.append(experiment)

    # =====================================
    # SAVE JSON
    # =====================================

    with open(
        EXP_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "✅ EXPERIMENT LOGGED"
    )