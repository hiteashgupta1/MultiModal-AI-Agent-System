import pandas as pd

df = pd.read_csv("experiments/metrics.csv")

print("Overall Confidence:", df["confidence"].mean())
print("Overall Latency:", df["latency"].mean())