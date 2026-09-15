import json
import os

import faiss
import numpy as np


VECTOR_STORE_PATH = "data/vector_store"


class VectorStore:
    """
    Stores document embeddings in FAISS
    and keeps the corresponding metadata.
    """

    def __init__(self, dimension):
        self.dimension = dimension

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.metadata = []

    def add_embeddings(self, embeddings, metadata):
        """
        Add embeddings and their metadata.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(embeddings)

        self.metadata.extend(metadata)

    def search(self, query_embedding, top_k=3):
        """
        Search for the most similar chunks.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        top_k = min(
            top_k,
            self.index.ntotal
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, index_id in zip(
            distances[0],
            indices[0]
        ):

            if index_id == -1:
                continue

            results.append({
                "distance": float(distance),
                "metadata": self.metadata[index_id]
            })

        return results

    def save(self, path=VECTOR_STORE_PATH):
        """
        Save FAISS index and metadata.
        """

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

        # Save FAISS index
        faiss.write_index(
            self.index,
            index_path
        )

        # Save metadata
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
            f"Vector store saved to: {path}"
        )

    def load(self, path=VECTOR_STORE_PATH):
        """
        Load FAISS index and metadata.
        """

        index_path = os.path.join(
            path,
            "index.faiss"
        )

        metadata_path = os.path.join(
            path,
            "metadata.json"
        )

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}"
            )

        # Load FAISS
        self.index = faiss.read_index(
            index_path
        )

        # Load metadata
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        print(
            f"Vector store loaded from: {path}"
        )