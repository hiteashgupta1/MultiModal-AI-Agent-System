import os
import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer



ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


MEMORY_DIR = os.path.join(
    ROOT_DIR,
    "backend",
    "rag"
)

os.makedirs(MEMORY_DIR, exist_ok=True)



INDEX_FILE = os.path.join(
    MEMORY_DIR,
    "memory.index"
)

TEXT_FILE = os.path.join(
    MEMORY_DIR,
    "memory_texts.json"
)


embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

dimension = 384



if os.path.exists(INDEX_FILE):

    index = faiss.read_index(INDEX_FILE)

else:

    index = faiss.IndexFlatL2(dimension)


if os.path.exists(TEXT_FILE):

    with open(
        TEXT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        memory_texts = json.load(f)

else:

    memory_texts = []


def store_memory(text):

    global memory_texts
    global index

    try:


        if text is None:
            return

        text = str(text).strip()

        if len(text) == 0:
            return


        embedding = embedder.encode([text])

        embedding = np.array(
            embedding,
            dtype="float32"
        )


        index.add(embedding)

        memory_texts.append(text)



        faiss.write_index(
            index,
            INDEX_FILE
        )



        with open(
            TEXT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                memory_texts,
                f,
                indent=2,
                ensure_ascii=False
            )


        print("\n✅ MEMORY STORED")

        print(
            "TEXT:",
            text[:100]
        )

        print(
            "TOTAL VECTORS:",
            index.ntotal
        )

    except Exception as e:

        print(
            "❌ MEMORY STORE ERROR:",
            str(e)
        )


def retrieve_memory(query, top_k=3):

    global memory_texts
    global index

    if len(memory_texts) == 0:
        return []

    query_embedding = embedder.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx, dist in zip(
        indices[0],
        distances[0]
    ):

        if idx < len(memory_texts):

            results.append({

                "text": memory_texts[idx],

                "distance": float(dist)

            })

    return results



def get_all_memory():

    global memory_texts

    return memory_texts[::-1]


def get_memory_count():

    global index

    return int(index.ntotal)