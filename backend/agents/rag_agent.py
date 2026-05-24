import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a lightweight embedding model perfect for CPU
model = SentenceTransformer("all-MiniLM-L6-v2")

# Default static fallback fallback text if no document is passed
DEFAULT_DOCUMENTS = [
    "Artificial Intelligence is the simulation of human intelligence.",
    "Machine Learning is a subset of AI."
]
default_embeddings = model.encode(DEFAULT_DOCUMENTS)
default_index = faiss.IndexFlatL2(default_embeddings.shape[1])
default_index.add(np.array(default_embeddings))


def retrieve_context(query, document_text=None):
    """
    Finds and returns sentences from the text matching the user query.
    Accepts both query and dynamic document text to resolve the TypeError.
    """
    # Case 1: Dynamic text from the uploaded PDF/Document is provided
    if document_text and document_text.strip():
        # Split the document text cleanly into sentences using regex
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', document_text) if len(s.strip()) > 8]
        
        if not sentences:
            return ""

        # Build a temporary runtime FAISS index for this specific document
        embeddings = model.encode(sentences)
        temp_index = faiss.IndexFlatL2(embeddings.shape[1])
        temp_index.add(np.array(embeddings))
        
        # Encode the user's question
        q_embed = model.encode([query])
        
        # Retrieve the top 3 most relevant sentences for better answer synthesis
        k = min(3, len(sentences))
        D, I = temp_index.search(np.array(q_embed), k=k)
        
        # Gather matching sentences, filtering out any empty slots (-1 index)
        retrieved_sentences = [sentences[idx] for idx in I[0] if idx != -1]
        return " ".join(retrieved_sentences)

    # Case 2: Fallback to global static list if no document text is supplied
    else:
        q_embed = model.encode([query])
        D, I = default_index.search(np.array(q_embed), k=1)
        return DEFAULT_DOCUMENTS[I[0][0]]