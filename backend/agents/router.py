from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()


def route_query(text=None, file_type=None):

    if file_type == "image":
        return "vision"

    if file_type == "pdf":
        return "summarizer"

    if text:
        text = text.lower()

        if any(k in text for k in ["draw image of", "generate image of", "create image of", "picture of"]):
            return "image_gen"

        if any(k in text for k in ["speak", "audio", "voice", "read this"]):
            return "tts"

        return "summarizer"

    return "summarizer"