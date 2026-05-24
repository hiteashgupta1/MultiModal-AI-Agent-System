import json
from datetime import datetime

def log_feedback(input_data, output_data, feedback):

    record = {
        "timestamp": str(datetime.now()),
        "input": input_data,
        "output": output_data,
        "feedback": feedback
    }

    with open("database/feedback.json", "a") as f:
        f.write(json.dumps(record) + "\n")