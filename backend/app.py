from fastapi import FastAPI, Request, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse, Response
import base64
import torch
from backend.agents.summarizer import summarize, tokenizer, model
from backend.agents.vision import detect_objects
from backend.agents.tts import text_to_speech
from backend.agents.image_gen import generate_image
from backend.agents.feedback import log_feedback
from backend.agents.orchestrator import smart_orchestrate
from backend.agents.router import route_query
from backend.agents.rag_agent import retrieve_context
import sys
import os
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pydantic import BaseModel
from feedback_store import save_feedback
import sqlite3


app = FastAPI()

# =========================
# 🚀 ROUTER (MAIN ENTRY)
# =========================
@app.post("/route")
async def route_api(
    request: Request,
    file: UploadFile = File(None),
    image: UploadFile = File(None)
):

    form = await request.form()
    text = form.get("text")

    file_type = None
    content = None

    # File upload
    if file:
        content = await file.read()
        if file.filename.endswith(".pdf"):
            file_type = "pdf"
        else:
            file_type = "image"

    # Camera image
    if image:
        content = await image.read()
        file_type = "image"

    agent = route_query(text, file_type)

    try:
        # =========================
        # 📝 SUMMARIZER
        # =========================
        if agent == "summarizer":
            result = summarize(input_text=text, pdf_bytes=content)

            return {
                "agent": agent,
                "output": result
            }

        # =========================
        # 🔍 VISION
        # =========================
        elif agent == "vision":
            result = detect_objects(content)

            # convert image → base64
            image_bytes = result.get("image")

            if isinstance(image_bytes, bytes):
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            else:
                image_b64 = image_bytes  # already encoded

            return {
                "agent": agent,
                "objects": result.get("objects", []),
                "image": image_b64
            }

        # =========================
        # 🎨 IMAGE GENERATION
        # =========================
        elif agent == "image_gen":
            result = generate_image(text)

            img = result.get("image") if isinstance(result, dict) else result

            if isinstance(img, bytes):
                img = base64.b64encode(img).decode("utf-8")

            return {
                "agent": agent,
                "image": img
            }

        # =========================
        # 🔊 TEXT TO SPEECH
        # =========================
        elif agent == "tts":
            lang = form.get("lang", "en")
            audio_bytes = text_to_speech(text, lang=lang)

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "agent": agent,
                "audio": audio_b64
            }

        return {"agent": "unknown"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# =========================
# 📝 SUMMARIZER API
# =========================
@app.post("/summarize")
async def summarize_api(
    text: str = Form(None),
    file: UploadFile = File(None)
):

    try:
        if file:
            pdf_bytes = await file.read()
            summary = summarize(pdf_bytes=pdf_bytes)

        elif text:
            summary = summarize(input_text=text)

        else:
            return {"error": "No input provided"}

        return summary

    except Exception as e:
        return {"error": str(e)}


# =========================
# 🔍 DETECT API
# =========================
@app.post("/detect")
async def detect_api(image: UploadFile = File(...)):

    try:
        image_bytes = await image.read()
        result = detect_objects(image_bytes)

        if not result:
            return {"error": "Detection failed"}

        # FIX: convert to base64 (NOT hex)
        image_b64 = base64.b64encode(result["image"]).decode("utf-8")

        return {
            "image": image_b64,
            "objects": result["objects"],
            "latency": result["latency"]
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# 🔊 TTS API
# =========================
@app.post("/text-to-speech")
async def tts(text: str):

    try:
        audio = text_to_speech(text)

        audio_b64 = base64.b64encode(audio).decode("utf-8")

        return {"audio": audio_b64}

    except Exception as e:
        return {"error": str(e)}


# =========================
# 🎨 IMAGE GEN API
# =========================
@app.post("/generate-image")
async def generate(prompt: str):

    try:
        img = generate_image(prompt)

        if isinstance(img, bytes):
            img = base64.b64encode(img).decode("utf-8")

        return {"image": img}

    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
async def analyze(
    text: str = Form(None),
    image: UploadFile = File(None),
    image_prompt: str = Form(None)
):

    image_bytes = None

    if image:
        image_bytes = await image.read()

    results, agents_used = orchestrate(
        text=text,
        image=image_bytes,
        generate_img_prompt=image_prompt
    )

    final_response = {
        "analysis": results,
        "agents_used": agents_used
    }

    return final_response


class FeedbackRequest(BaseModel):
    input_text: str | None = None
    rating: int
    comment: str | None = None
    agent: str
@app.post("/feedback")
async def submit_feedback(data: FeedbackRequest):

    # If empty, replace with placeholder
    input_text = data.input_text if data.input_text else "N/A"
    comment = data.comment if data.comment else ""

    # Log feedback (your existing function)
    save_feedback(input_text, data.rating, comment)

    return {"status": "Feedback saved successfully"}

# --- ADD THIS ENDPOINT TO YOUR FASTAPI BACKEND ---

class QARequest(BaseModel):
    question: str
    document_text: str

@app.post("/qa")
async def qa_document(data: QARequest):
    question = data.question
    doc_text = data.document_text

    if not doc_text.strip():
        return {"answer": "Please upload a document first before asking questions.", "sources": ""}

    # 1. Fetch exact matching sentences from the original text using your RAG agent
    retrieved_sources = retrieve_context(question, doc_text)

    if not retrieved_sources or len(retrieved_sources.strip()) < 5:
        return {
            "answer": "I couldn't find any relevant sentences in the document to answer your question.",
            "sources": "No direct matches found."
        }

    # 2. Use your loaded model (FLAN-T5) to generate a natural answer using the source context
    prompt = f"context: {retrieved_sources} question: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            num_beams=3, 
            early_stopping=True
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "answer": answer,
        "sources": retrieved_sources  # Sending back the exact retrieved sentences
    }