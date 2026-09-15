import json
import os

import faiss
import numpy as np


# ==================================================
# CONFIGURATION
# ==================================================

VISUAL_VECTOR_STORE_PATH = (
    "data/visual_vector_store"
)


# ==================================================
# VISUAL VECTOR STORE
# ==================================================

class VisualVectorStore:
    """
    Stores visual embeddings in FAISS and keeps
    the corresponding visual metadata.
    """

    def __init__(self, dimension):

        self.dimension = dimension

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.metadata = []


    # ==================================================
    # ADD EMBEDDINGS
    # ==================================================

    def add_embeddings(
        self,
        embeddings,
        metadata
    ):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:

            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:

            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        if len(embeddings) != len(metadata):

            raise ValueError(
                "Number of embeddings must match "
                "number of metadata records."
            )

        self.index.add(
            embeddings
        )

        self.metadata.extend(
            metadata
        )


    # ==================================================
    # SEARCH
    # ==================================================

    def search(
        self,
        query_embedding,
        top_k=3
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:

            query_embedding = (
                query_embedding.reshape(
                    1,
                    -1
                )
            )

        if query_embedding.shape[1] != self.dimension:

            raise ValueError(
                f"Query embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {query_embedding.shape[1]}."
            )

        if self.index.ntotal == 0:

            return []

        top_k = min(
            top_k,
            self.index.ntotal
        )

        distances, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for distance, index_id in zip(
            distances[0],
            indices[0]
        ):

            if index_id == -1:
                continue

            results.append({

                "distance":
                    float(distance),

                "metadata":
                    self.metadata[index_id]
            })

        return results


    # ==================================================
    # SAVE
    # ==================================================

    def save(
        self,
        path=VISUAL_VECTOR_STORE_PATH
    ):

        os.makedirs(
            path,
            exist_ok=True
        )

        index_path = os.path.join(
            path,
            "index.faiss"
        )

        metadata_path = os.path.join(
            path,
            "metadata.json"
        )

        faiss.write_index(
            self.index,
            index_path
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"Visual vector store saved to: "
            f"{path}"
        )

        print(
            f"Visual vectors saved: "
            f"{self.index.ntotal}"
        )


    # ==================================================
    # LOAD
    # ==================================================

    def load(
        self,
        path=VISUAL_VECTOR_STORE_PATH
    ):

        index_path = os.path.join(
            path,
            "index.faiss"
        )

        metadata_path = os.path.join(
            path,
            "metadata.json"
        )

        if not os.path.exists(
            index_path
        ):

            raise FileNotFoundError(
                f"Visual FAISS index not found: "
                f"{index_path}"
            )

        if not os.path.exists(
            metadata_path
        ):

            raise FileNotFoundError(
                f"Visual metadata not found: "
                f"{metadata_path}"
            )

        self.index = faiss.read_index(
            index_path
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(
                file
            )

        if self.index.ntotal != len(
            self.metadata
        ):

            raise ValueError(
                "Visual vector store is inconsistent: "
                f"{self.index.ntotal} vectors but "
                f"{len(self.metadata)} metadata records."
            )

        print(
            f"Visual vector store loaded from: "
            f"{path}"
        )

        print(
            f"Visual vectors available: "
            f"{self.index.ntotal}"
        )
# ==================================================
# BUILD VISUAL VECTOR STORE
# ==================================================

from sentence_transformers import SentenceTransformer


VISUAL_METADATA_PATH = (
    "data/visual_analysis/"
    "visual_metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_visual_metadata():

    if not os.path.exists(
        VISUAL_METADATA_PATH
    ):

        raise FileNotFoundError(
            f"Visual metadata not found: "
            f"{VISUAL_METADATA_PATH}"
        )

    with open(
        VISUAL_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_search_text(item):

    return (
        f"Source: {item.get('source', '')}. "
        f"Page: {item.get('page', '')}. "
        f"Image: {item.get('image', '')}. "
        f"Visual analysis: "
        f"{item.get('analysis', '')}"
    )


def build_visual_vector_store():

    print(
        "\n===================================="
    )

    print(
        "BUILDING VISUAL VECTOR STORE"
    )

    print(
        "===================================="
    )

    # ------------------------------------------------
    # Load metadata
    # ------------------------------------------------

    metadata = load_visual_metadata()

    print(
        f"Visual metadata records: "
        f"{len(metadata)}"
    )

    if not metadata:

        raise ValueError(
            "No visual metadata found."
        )

    # ------------------------------------------------
    # Load embedding model
    # ------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # ------------------------------------------------
    # Build searchable text
    # ------------------------------------------------

    texts = []

    for item in metadata:

        texts.append(
            build_search_text(
                item
            )
        )

    # ------------------------------------------------
    # Create embeddings
    # ------------------------------------------------

    print(
        "\nBuilding visual embeddings..."
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # ------------------------------------------------
    # Create vector store
    # ------------------------------------------------

    dimension = embeddings.shape[1]

    vector_store = VisualVectorStore(
        dimension
    )

    # ------------------------------------------------
    # Add embeddings + metadata
    # ------------------------------------------------

    vector_store.add_embeddings(
        embeddings,
        metadata
    )

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    vector_store.save(
        VISUAL_VECTOR_STORE_PATH
    )

    print(
        "\n===================================="
    )

    print(
        "VISUAL VECTOR STORE READY"
    )

    print(
        "===================================="
    )

    print(
        f"Vectors : "
        f"{vector_store.index.ntotal}"
    )

    print(
        f"Metadata: "
        f"{len(vector_store.metadata)}"
    )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    build_visual_vector_store()