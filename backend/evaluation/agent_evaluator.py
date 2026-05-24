import pandas as pd

def evaluate_summary(summary, confidence, hallucination):

    score = 0

    if confidence > 0.7:
        score += 1

    if not hallucination:
        score += 1

    if len(summary.split()) > 20:
        score += 1

    return score