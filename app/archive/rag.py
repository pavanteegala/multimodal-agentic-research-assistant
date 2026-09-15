import os

import numpy as np
from dotenv import load_dotenv
from google import genai

from app.document_processor import process_pdf
from app.embedder import create_embeddings
from app.vector_store import VectorStore


PDF_PATH = "data/documents/multimodal_research_paper.pdf"

TOP_K = 2

VECTOR_STORE_PATH = "data/vector_store"


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file"
    )


# --------------------------------------------------
# Create Gemini client
# --------------------------------------------------

client = genai.Client(
    api_key=api_key
)


# --------------------------------------------------
# Create or load vector store
# --------------------------------------------------

index_path = os.path.join(
    VECTOR_STORE_PATH,
    "index.faiss"
)

metadata_path = os.path.join(
    VECTOR_STORE_PATH,
    "metadata.json"
)


if os.path.exists(index_path) and os.path.exists(metadata_path):

    # ----------------------------------------------
    # Load existing vector store
    # ----------------------------------------------

    print("\nLoading existing vector store...")

    vector_store = VectorStore(
        dimension=384
    )

    vector_store.load(
        VECTOR_STORE_PATH
    )

    print(
        f"Vectors available: "
        f"{vector_store.index.ntotal}"
    )

else:

    # ----------------------------------------------
    # Build vector store for the first time
    # ----------------------------------------------

    print("\nBuilding vector store...")

    chunks = process_pdf(
        PDF_PATH
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Create embeddings
    embeddings = create_embeddings(
        texts
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # Create vector store
    dimension = embeddings.shape[1]

    vector_store = VectorStore(
        dimension
    )

    # Add vectors + metadata
    vector_store.add_embeddings(
        embeddings,
        chunks
    )

    # Save
    vector_store.save(
        VECTOR_STORE_PATH
    )


# --------------------------------------------------
# Ask question
# --------------------------------------------------

query = input(
    "\nAsk a question about the PDF: "
)


# --------------------------------------------------
# Create query embedding
# --------------------------------------------------

query_embedding = create_embeddings(
    [query]
)

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)


# --------------------------------------------------
# Search
# --------------------------------------------------

results = vector_store.search(
    query_embedding,
    top_k=TOP_K
)


# --------------------------------------------------
# Display retrieved sources
# --------------------------------------------------

print("\n====================================")
print("RETRIEVED SOURCES")
print("====================================")


context_parts = []


for result in results:

    metadata = result["metadata"]

    distance = result["distance"]

    print(
        f"- {metadata['source']} "
        f"(Page {metadata['page']}, "
        f"Chunk {metadata['chunk_id']}, "
        f"Distance {distance:.4f})"
    )

    context_parts.append(
        metadata["text"]
    )


context = "\n\n".join(
    context_parts
)


# --------------------------------------------------
# Create grounded prompt
# --------------------------------------------------

prompt = f"""
You are a research assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer cannot be found in the context,
say that the information is not available
in the document.

Do not invent facts.

Context:
--------------------
{context}
--------------------

Question:
{query}

Provide a clear and concise answer.
"""


# --------------------------------------------------
# Ask Gemini
# --------------------------------------------------

response = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)


# --------------------------------------------------
# Display answer
# --------------------------------------------------

print("\n====================================")
print("ANSWER")
print("====================================")

print(
    response.output_text
)