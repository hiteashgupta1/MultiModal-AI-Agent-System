import pandas as pd
import os

def calibrate(agent, confidence):

    if not os.path.exists("experiments/feedback.csv"):
        return confidence

    fb = pd.read_csv("experiments/feedback.csv")

    agent_fb = fb[fb["agent"] == agent]

    if len(agent_fb) < 5:
        return confidence

    avg_rating = agent_fb["rating"].mean()

    adjusted = 0.7 * confidence + 0.3 * (avg_rating / 5)

    return round(adjusted, 2)