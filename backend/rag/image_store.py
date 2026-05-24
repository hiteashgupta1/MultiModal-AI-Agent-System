import os
import json
import faiss
import numpy as np

from PIL import Image
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
    "image_memory"
)

os.makedirs(
    MEMORY_DIR,
    exist_ok=True
)


INDEX_FILE = os.path.join(
    MEMORY_DIR,
    "image.index"
)

META_FILE = os.path.join(
    MEMORY_DIR,
    "image_meta.json"
)

# =========================================
# LOAD CLIP MODEL
# =========================================

model = SentenceTransformer(
    "sentence-transformers/clip-ViT-B-32"
)

dimension = 512

# =========================================
# LOAD OR CREATE FAISS INDEX
# =========================================

if os.path.exists(INDEX_FILE):

    index = faiss.read_index(INDEX_FILE)

else:

    index = faiss.IndexFlatL2(dimension)

# =========================================
# LOAD METADATA
# =========================================

if os.path.exists(META_FILE):

    with open(
        META_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

else:

    metadata = []

# =========================================
# STORE PERSON IMAGE
# =========================================

def store_person_image(image_path):

    global metadata
    global index

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        embedding = model.encode(image)

        embedding = np.array(
            [embedding],
            dtype="float32"
        )

        # =================================
        # ADD TO FAISS
        # =================================

        index.add(embedding)

        # =================================
        # SAVE METADATA
        # =================================

        metadata.append({

            "path": image_path

        })



        faiss.write_index(
            index,
            INDEX_FILE
        )


        with open(
            META_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            "✅ IMAGE STORED:",
            image_path
        )

        print(
            "TOTAL IMAGE VECTORS:",
            index.ntotal
        )

    except Exception as e:

        print(
            "❌ IMAGE STORE ERROR:",
            str(e)
        )

def search_similar_images(
    query_image_path,
    top_k=5
):

    global metadata
    global index

    try:

        image = Image.open(
            query_image_path
        ).convert("RGB")

        embedding = model.encode(image)

        embedding = np.array(
            [embedding],
            dtype="float32"
        )

        distances, indices = index.search(
            embedding,
            top_k
        )

        results = []

        for idx in indices[0]:

            if idx < len(metadata):

                results.append(
                    metadata[idx]["path"]
                )

        return results

    except Exception as e:

        print(
            "❌ IMAGE SEARCH ERROR:",
            str(e)
        )

        return []


def get_image_memory_count():

    return int(index.ntotal)