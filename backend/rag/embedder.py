from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def hallucination_score(text, summary):

    emb1 = model.encode([text])
    emb2 = model.encode([summary])

    score = cosine_similarity(emb1, emb2)[0][0]

    return score