from sentence_transformers import SentenceTransformer
import threading

_model = None
_lock = threading.Lock()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_model():
    global _model

    if _model is None:
        with _lock:
            if _model is None:
                print("Loading embedding model...")
                _model = SentenceTransformer(
                    MODEL_NAME,
                    device="cpu"
                )

    return _model


def retrieve_context(text):
    model = get_model()

    embedding = model.encode(text)

    return embedding