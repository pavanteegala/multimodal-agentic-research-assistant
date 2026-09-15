from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(texts):
    """
    Convert a list of text strings
    into numerical embeddings.
    """

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


if __name__ == "__main__":

    sample_texts = [
        "Artificial intelligence is a field of computer science.",
        "Machine learning allows computers to learn from data.",
        "Retrieval augmented generation combines retrieval with language models."
    ]

    embeddings = create_embeddings(sample_texts)

    print("Number of texts:", len(sample_texts))

    print("Embedding shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])