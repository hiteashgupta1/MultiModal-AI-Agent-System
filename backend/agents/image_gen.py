import requests
import base64
import time
from backend.evaluation.log_metrics import log_metric
from backend.experiments.experiment_tracker import log_experiment

COLAB_URL = "https://galeiform-cathleen-unloveably.ngrok-free.dev"

def generate_image(prompt):
    start = time.time()
    latency = 0.0
    result_image = None

    try:
        response = requests.post(
            f"{COLAB_URL}/generate",
            json={"prompt": prompt}
        )

        if response.status_code == 200:
            result_image = base64.b64encode(response.content).decode("utf-8")
        else:
            print("COLAB ERROR:", response.text)
            return None, 0.1

    except Exception as e:
        print("COLAB CONNECTION ERROR:", str(e))
        return None, 0.1
    finally:
        latency = round(time.time() - start, 2)

        try:
            log_metric(
                agent="image_gen",
                model_version="img_v1",
                confidence=0.99,
                object_count=None,
                latency=latency,
                hallucination=False
            )

            log_experiment(
                agent="image_gen",
                confidence=0.99,
                latency=latency,
                hallucination=False,
                model_version="img_v1"
            )
        except Exception as log_err:
            print("Metric logging failed:", log_err)

    return result_image, 0.85