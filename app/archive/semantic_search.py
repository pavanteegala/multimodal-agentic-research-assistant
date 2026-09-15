from app.document_processor import process_pdf
from app.embedder import create_embeddings
import faiss
import numpy as np


PDF_PATH = "data/documents/research_paper.pdf"


# --------------------------------------------------
# 1. Process the PDF
# --------------------------------------------------

chunks = process_pdf(PDF_PATH)

print(f"Total chunks: {len(chunks)}")


# --------------------------------------------------
# 2. Extract the text from each chunk
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embeddings = create_embeddings(texts)

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# 4. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.asarray(embeddings, dtype="float32")
)

print(f"Vectors stored in FAISS: {index.ntotal}")


# --------------------------------------------------
# 5. Ask a question
# --------------------------------------------------

query = input("\nAsk a question about the PDF: ")


# --------------------------------------------------
# 6. Convert question into an embedding
# --------------------------------------------------

query_embedding = create_embeddings([query])

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)


# --------------------------------------------------
# 7. Search FAISS
# --------------------------------------------------

top_k = min(3, len(chunks))

distances, indices = index.search(
    query_embedding,
    top_k
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n========== SEARCH RESULTS ==========")

for rank, (distance, index_id) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):

    chunk = chunks[index_id]

    print(f"\nResult {rank}")
    print("-----------------------------")

    print(f"Chunk ID : {chunk['chunk_id']}")
    print(f"Source   : {chunk['source']}")
    print(f"Page     : {chunk['page']}")
    print(f"Distance : {distance:.4f}")

    print("\nText:")
    print(chunk["text"][:700])