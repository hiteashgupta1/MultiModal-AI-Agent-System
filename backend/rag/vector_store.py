import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
index = faiss.IndexFlatL2(384)

def add_document(text):

    embedding = model.encode([text])

    index.add(np.array(embedding).astype("float32"))

    documents.append(text)


def retrieve_context(query):

    embedding = model.encode([query])

    D, I = index.search(np.array(embedding).astype("float32"), k=3)

    results = [documents[i] for i in I[0] if i < len(documents)]

    return "\n".join(results)