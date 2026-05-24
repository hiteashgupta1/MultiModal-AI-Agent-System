from backend.agents.summarizer import summarize
from backend.agents.vision import detect_objects
from backend.evaluation.calibration import calibrate

def smart_orchestrate(text=None, image=None):

    results = {}
    confidences = []
    agent_used = []

    # Text agent
    if text:
        summary, conf = summarize(text)
        conf = calibrate("summarizer", conf)

        results["summary"] = summary
        confidences.append(conf)
        agent_used.append("summarizer")

    # Vision agent
    if image:
        vision_result = detect_objects(image)

        if vision_result:
            avg_conf = sum(o["confidence"] for o in vision_result["objects"]) / max(len(vision_result["objects"]), 1)
            avg_conf = calibrate("vision", avg_conf)

            results["objects"] = vision_result["objects"]
            results["image"] = vision_result["image"]
            confidences.append(avg_conf)
            agent_used.append("vision")

    overall_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return {
        "results": results,
        "confidence": overall_conf,
        "agents_used": agent_used
    }