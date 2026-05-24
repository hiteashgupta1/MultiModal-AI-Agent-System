import os
import torch
import fitz
import time
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from backend.rag.memory_store import store_memory, retrieve_memory
from backend.agents.rag_agent import retrieve_context
from backend.evaluation.log_metrics import log_metric
from backend.experiments.experiment_tracker import log_experiment
from backend.evaluation.summarization_metrics import compute_rouge, compute_bertscore
import numpy as np

import logging
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.getLogger("transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Downloaded")

# --- INITIALIZATION ---
BASE_MODEL = "google/flan-t5-small" 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def get_latest_model():
    versions = []
    if not os.path.exists(MODEL_DIR): return None
    for name in os.listdir(MODEL_DIR):
        if name.startswith("orchestrator_v"):
            try:
                v = int(name.replace("orchestrator_v", ""))
                versions.append(v)
            except: pass
    if not versions: return None
    latest = max(versions)
    return os.path.join(MODEL_DIR, f"orchestrator_v{latest}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
ADAPTER_PATH = get_latest_model()

if ADAPTER_PATH and os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    except:
        model = base_model
else:
    model = base_model
model.eval()

# --- HELPERS ---

def extract_pdf_text(pdf_bytes):
    text = ""
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def get_chunks(text, chunk_size=600):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def detect_hallucination(original_text, summary):
    original_words = set(re.sub(r'[^\w\s]', '', original_text.lower()).split())
    summary_words = re.sub(r'[^\w\s]', '', summary.lower()).split()
    if not summary_words: return False, 0.0
    unseen_words = [w for w in summary_words if w not in original_words]
    ratio = len(unseen_words) / len(summary_words)
    return ratio > 0.3, round(ratio, 2)

def calculate_confidence(original_text, summary):
    orig_len = len(original_text.split())
    sum_len = len(summary.split())
    if orig_len == 0 or sum_len == 0: return 0.1
    ratio = sum_len / orig_len
    score = 1 - abs(ratio - 0.3)
    return round(max(0.1, min(0.6 + 0.4 * score, 1.0)), 2)

# --- CORE LOGIC ---
def clean_input_text(text: str) -> str:
    """Removes noise like [1], [2], [105] and extra whitespaces."""
    if not text:
        return ""
    # 1. Remove bracketed numerical citations like [1], [22], [105]
    text = re.sub(r'\[\d+\]', '', text)
    # 2. Clean up multiple spaces left behind by the removal
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def summarize(input_text=None, pdf_bytes=None):
    start = time.time()
    text = extract_pdf_text(pdf_bytes) if pdf_bytes else input_text

    if not text or len(text.strip()) < 50:
        return {"summary": "Input text is too short to provide a meaningful summary.", "confidence": 0.0}

    # 1. MAP PHASE: Deep Scanning
    chunks = list(get_chunks(text))
    chunk_summaries = []
    
    for chunk in chunks:
        # Instruction tells model to ignore metadata/citations
        prompt = f"summarize the main findings and technical details from this section, skipping any titles, authors, or journal citations: {chunk}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=4,
                repetition_penalty=2.5,
                early_stopping=True
            )
        
        chunk_summaries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # 2. REDUCE PHASE: Whole-Document Synthesis
    if len(chunk_summaries) > 1:
        combined_text = " ".join(chunk_summaries)
        # Force a structured beginning-to-end overview
        prompt = (
            "Write a comprehensive summary of the entire document based on these sections. "
            "Start with the primary objective, explain the methodology/findings, and end with the conclusion. "
            f"Sections: {combined_text}"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=400, # More space for a "starting to end" summary
                min_length=150,     # Prevent it from being too short
                num_beams=5,        # Higher quality search
                no_repeat_ngram_size=3
            )
        input_type = "PDF Document" if pdf_bytes else "Raw Text Input"
        final_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        final_summary = chunk_summaries[0] if chunk_summaries else "Summary generation failed."

    # Final Polish: Fix ending sentences
    if not final_summary.endswith('.'):
        last_period = final_summary.rfind('.')
        if last_period != -1:
            final_summary = final_summary[:last_period+1]
    store_memory(text)
    store_memory(final_summary)
    # --- METRICS ---
    confidence = calculate_confidence(text, final_summary)
    hallucinated, ratio = detect_hallucination(text, final_summary)
    
    rouge = compute_rouge(text[:1000], final_summary)
    bert = compute_bertscore(text[:1000], final_summary)
    latency = round(time.time() - start, 2)

    log_experiment(
        agent="summarizer",
        confidence=confidence,
        latency=latency,
        hallucination=hallucinated,
        model_version="summarizer_v1"
    )

    log_metric(agent="summarizer", model_version="summary_v1", confidence=float(confidence), latency=latency, hallucination=hallucinated)

    return {
        "summary": final_summary,
        "confidence": float(confidence),
        "hallucinated": bool(hallucinated),
        "hallucination_ratio": float(ratio),
        "rouge1": float(rouge.get("rouge1", 0)),
        "rouge2": float(rouge.get("rouge2", 0)),
        "rougeL": float(rouge.get("rougeL", 0)),
        "bertscore_precision": float(bert.get("bertscore_precision", 0)),
        "bertscore_recall": float(bert.get("bertscore_recall", 0)),
        "bertscore_f1": float(bert.get("bertscore_f1", 0))
    }