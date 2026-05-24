import pandas as pd
import os

THRESHOLD = 0.55

def check_performance():

    if not os.path.exists("experiments/metrics.csv"):
        return False

    df = pd.read_csv("experiments/metrics.csv")

    if len(df) < 5:
        return False

    recent = df.tail(5)
    avg_conf = recent["avg_confidence"].mean()

    print("Recent average confidence:", avg_conf)

    if avg_conf < THRESHOLD:
        return True

    return False

def moving_average(series, window=5):
    return series.rolling(window=window).mean()


def should_retrain():

    if not os.path.exists("experiments/feedback.csv"):
        return False

    fb = pd.read_csv("experiments/feedback.csv")

    if len(fb) < 10:
        return False

    recent = fb.tail(10)

    return recent["rating"].mean() < 3.0