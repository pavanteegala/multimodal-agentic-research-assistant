def create_chunks(text, chunk_size=500, overlap=80):
    """
    Split text into overlapping chunks.

    chunk_size:
        Maximum number of words in each chunk.

    overlap:
        Number of words shared between consecutive chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start = end - overlap

    return chunks

if __name__ == "__main__":

    sample_text = """
    Artificial intelligence is a field of computer science.
    Machine learning allows computers to learn patterns from data.
    Retrieval augmented generation combines information retrieval
    with large language models to improve factual responses.
    Embeddings represent text as numerical vectors.
    Vector databases allow us to search these representations.
    """

    chunks = create_chunks(
        sample_text,
        chunk_size=20,
        overlap=5
    )

    for i, chunk in enumerate(chunks):

        print(f"\n--- CHUNK {i + 1} ---")
        print(chunk)