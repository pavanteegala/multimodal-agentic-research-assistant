"""
Multimodal RAG backend.

This module connects:
    - PDF text processing
    - text vector retrieval
    - visual vector retrieval
    - query classification
    - Gemini answer generation
    - the agent orchestration layer
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai

from app.agent import ResearchAgent
from app.config import (
    DEFAULT_PDF_PATH,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODEL,
    TEXT_TOP_K,
    VISUAL_TOP_K,
    VISUAL_VECTOR_STORE_DIR,
    VECTOR_STORE_DIR,
)
from app.document_processor import process_pdf
from app.embedder import create_embeddings
from app.vector_store import VectorStore
from app.visual_vector_store import VisualVectorStore


# ============================================================
# Environment / Gemini client
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Add your Gemini API key to the .env file."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# Compatibility aliases
# ============================================================

# These aliases keep the original names available to other
# parts of the project.
PDF_PATH = str(DEFAULT_PDF_PATH)

TEXT_VECTOR_STORE_PATH = str(VECTOR_STORE_DIR)

VISUAL_VECTOR_STORE_PATH = str(
    VISUAL_VECTOR_STORE_DIR
)

EMBEDDING_DIM = EMBEDDING_DIMENSION


# ============================================================
# Utility helpers
# ============================================================

def _get_text(item: Any) -> str:
    """
    Extract text from either:
        - a plain string
        - a dictionary containing a 'text' field
    """

    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        return str(
            item.get(
                "text",
                "",
            )
        )

    return str(item)


def _normalize(value: Any) -> str:
    """
    Normalize text for lightweight keyword matching.
    """

    return re.sub(
        r"\s+",
        " ",
        str(value).lower(),
    ).strip()


def _tokens(value: Any) -> set[str]:
    """
    Convert text into simple alphanumeric tokens.
    """

    return set(
        re.findall(
            r"[a-z0-9]+",
            _normalize(value),
        )
    )


def _keyword_overlap(
    query: str,
    text: str,
) -> float:
    """
    Calculate lightweight lexical overlap.

    This complements FAISS semantic similarity and helps
    with exact terms such as names, regions, materials,
    and values.
    """

    query_tokens = _tokens(query)
    text_tokens = _tokens(text)

    if not query_tokens or not text_tokens:
        return 0.0

    return len(
        query_tokens & text_tokens
    ) / len(query_tokens)


def _safe_distance_score(
    distance: float,
) -> float:
    """
    Convert FAISS L2 distance into a descending
    relevance score.
    """

    if distance < 0:
        distance = 0.0

    return 1.0 / (1.0 + distance)


# ============================================================
# Query classification
# ============================================================

def detect_query_type(
    query: str,
) -> str:
    """
    Classify a question as mainly text-oriented
    or visual-oriented.

    The visual keyword list includes terms commonly
    associated with figures, charts, tables, graphs,
    visual interpretation, and chart-based comparisons.
    """

    query_normalized = _normalize(query)

    visual_keywords = {
        "image",
        "images",
        "figure",
        "figures",
        "chart",
        "charts",
        "graph",
        "graphs",
        "table",
        "tables",
        "visual",
        "visuals",
        "diagram",
        "diagrams",
        "plot",
        "plots",
        "bar",
        "bars",
        "axis",
        "axes",
        "percentage",
        "percent",
        "share",
        "shown",
        "shows",
        "pictured",
        "illustration",
        "illustrations",
        "woven",
        "knitted",
        "velours",
        "material",
        "materials",
        "grafts",
        "graft",
        "adoption",
        "adoptions",
        "region",
        "regions",
        "highest",
        "lowest",
    }

    tokens = set(
        query_normalized.split()
    )

    if tokens & visual_keywords:
        return "visual"

    return "text"


# ============================================================
# Load text vector store
# ============================================================

def load_text_vector_store() -> VectorStore:
    """
    Load the default text vector store.

    If it does not exist, build it from the default PDF.
    """

    vector_store = VectorStore(
        dimension=EMBEDDING_DIMENSION
    )

    index_path = os.path.join(
        TEXT_VECTOR_STORE_PATH,
        "index.faiss",
    )

    metadata_path = os.path.join(
        TEXT_VECTOR_STORE_PATH,
        "metadata.json",
    )

    if (
        os.path.exists(index_path)
        and os.path.exists(metadata_path)
    ):
        vector_store.load(
            TEXT_VECTOR_STORE_PATH
        )

        return vector_store

    print(
        "Text vector store not found."
    )

    print(
        f"Building from: {PDF_PATH}"
    )

    chunks = process_pdf(
        PDF_PATH
    )

    if not chunks:
        raise ValueError(
            "No text chunks were extracted "
            f"from: {PDF_PATH}"
        )

    texts = [
        _get_text(chunk)
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        texts
    )

    metadata = []

    for index, chunk in enumerate(
        chunks
    ):
        if isinstance(chunk, dict):
            item = dict(chunk)
        else:
            item = {
                "text": str(chunk),
                "chunk_id": index + 1,
            }

        metadata.append(item)

    vector_store.add_embeddings(
        embeddings,
        metadata,
    )

    vector_store.save(
        TEXT_VECTOR_STORE_PATH
    )

    return vector_store


# ============================================================
# Load visual vector store
# ============================================================

def load_visual_vector_store() -> VisualVectorStore:
    """
    Load the default visual vector store.
    """

    visual_store = VisualVectorStore(
        dimension=EMBEDDING_DIMENSION
    )

    visual_store.load(
        VISUAL_VECTOR_STORE_PATH
    )

    return visual_store


# ============================================================
# Text search
# ============================================================

def search_text(
    query: str,
    query_embedding: np.ndarray,
    text_store: VectorStore,
    top_k: int = TEXT_TOP_K,
) -> list[dict]:
    """
    Search text evidence using semantic similarity
    plus lightweight keyword overlap.
    """

    results = text_store.search(
        query_embedding,
        top_k=top_k,
    )

    ranked_results = []

    for result in results:
        metadata = result.get(
            "metadata",
            {},
        )

        text = _get_text(
            metadata
        )

        semantic_score = (
            _safe_distance_score(
                float(
                    result.get(
                        "distance",
                        0.0,
                    )
                )
            )
        )

        keyword_score = (
            _keyword_overlap(
                query,
                text,
            )
        )

        combined_score = (
            0.70 * semantic_score
            + 0.30 * keyword_score
        )

        ranked_results.append(
            {
                **result,
                "score": combined_score,
            }
        )

    ranked_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked_results


# ============================================================
# Visual search
# ============================================================

def search_visual(
    query: str,
    query_embedding: np.ndarray,
    visual_store: VisualVectorStore,
    top_k: int = VISUAL_TOP_K,
) -> list[dict]:
    """
    Search visual evidence using visual analysis
    and image metadata.
    """

    results = visual_store.search(
        query_embedding,
        top_k=top_k,
    )

    ranked_results = []

    for result in results:
        metadata = result.get(
            "metadata",
            {},
        )

        if isinstance(metadata, dict):
            analysis = metadata.get(
                "analysis",
                "",
            )

            image_name = metadata.get(
                "image",
                "",
            )

            combined_text = (
                f"{analysis} {image_name}"
            )
        else:
            combined_text = str(
                metadata
            )

        semantic_score = (
            _safe_distance_score(
                float(
                    result.get(
                        "distance",
                        0.0,
                    )
                )
            )
        )

        keyword_score = (
            _keyword_overlap(
                query,
                combined_text,
            )
        )

        combined_score = (
            0.70 * semantic_score
            + 0.30 * keyword_score
        )

        ranked_results.append(
            {
                **result,
                "score": combined_score,
            }
        )

    ranked_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked_results


# ============================================================
# Query embedding
# ============================================================

def create_query_embedding(
    query: str,
) -> np.ndarray:
    """
    Create one embedding for the user's question.
    """

    embedding = create_embeddings(
        [query]
    )[0]

    embedding = np.asarray(
        embedding,
        dtype="float32",
    )

    if embedding.ndim == 1:
        embedding = embedding.reshape(
            1,
            -1,
        )

    if (
        embedding.shape[1]
        != EMBEDDING_DIMENSION
    ):
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{embedding.shape[1]}. "
            f"Expected {EMBEDDING_DIMENSION}."
        )

    return embedding


# ============================================================
# Context building
# ============================================================

def build_context(
    query: str,
    query_type: str,
    text_results: list[dict],
    visual_results: list[dict],
) -> str:
    """
    Build a compact context for the answer generator.

    Visual questions prioritize visual evidence.
    Text questions prioritize text evidence.
    """

    sections: list[str] = []

    if query_type == "visual":
        selected_visual = (
            visual_results[:1]
        )

        selected_text = (
            text_results[:1]
        )
    else:
        selected_text = (
            text_results[:1]
        )

        selected_visual = (
            visual_results[:1]
        )

    # --------------------------------------------------------
    # Visual evidence
    # --------------------------------------------------------

    for index, result in enumerate(
        selected_visual,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {
                "analysis": str(metadata)
            }

        source = metadata.get(
            "source",
            "unknown source",
        )

        page = metadata.get(
            "page",
            "unknown",
        )

        image = metadata.get(
            "image",
            metadata.get(
                "filename",
                "unknown image",
            ),
        )

        analysis = metadata.get(
            "analysis",
            "",
        )

        sections.append(
            f"[Visual Evidence V{index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Image: {image}\n"
            f"Analysis: {analysis}"
        )

    # --------------------------------------------------------
    # Text evidence
    # --------------------------------------------------------

    for index, result in enumerate(
        selected_text,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {
                "text": str(metadata)
            }

        source = metadata.get(
            "source",
            "unknown source",
        )

        page = metadata.get(
            "page",
            "unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            "unknown",
        )

        text = metadata.get(
            "text",
            "",
        )

        sections.append(
            f"[Text Evidence T{index}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk_id}\n"
            f"Text: {text}"
        )

    if not sections:
        return (
            "No relevant evidence was retrieved."
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# Gemini error detection
# ============================================================

def _is_quota_or_rate_limit_error(
    error: Exception,
) -> bool:
    """
    Detect common Gemini quota / rate-limit failures
    without depending on a specific SDK exception class.
    """

    error_text = str(
        error
    ).lower()

    quota_terms = {
        "429",
        "quota",
        "rate limit",
        "ratelimit",
        "resource exhausted",
        "too many requests",
        "free_tier_requests",
    }

    return any(
        term in error_text
        for term in quota_terms
    )


# ============================================================
# Gemini answer generation
# ============================================================

def generate_answer(
    query: str,
    query_type: str,
    context: str,
) -> str:
    """
    Generate a grounded answer using Gemini.

    Gemini failures are converted into user-friendly
    messages instead of exposing a Python traceback.
    """

    if not context.strip():
        return (
            "I could not find relevant evidence in "
            "the document for this question."
        )

    # Multi-hop prompts are already prepared by the
    # ResearchAgent.
    if query_type == "multi_hop":
        prompt = context

    else:
        prompt = f"""
You are a document research assistant.

Answer the user's question using ONLY the supplied evidence.

User question:
{query}

Evidence:
{context}

Instructions:
1. Answer directly and clearly.
2. Do not invent facts.
3. Do not use outside knowledge.
4. Keep the answer concise unless explanation is necessary.
5. Base every factual statement on the supplied evidence.
"""

    try:
        response = client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt,
        )

        # Gemini Interactions API returns generated
        # answer text through `output_text`.
        answer = getattr(
            response,
            "output_text",
            None,
        )

        # Compatibility fallback for SDK/model variants.
        if not answer:
            answer = getattr(
                response,
                "text",
                None,
            )

        if not answer:
            return (
                "The AI service completed the request, "
                "but did not return readable answer text."
            )

        return str(
            answer
        ).strip()

    except Exception as error:
        print(
            "Gemini generation failed: "
            f"{type(error).__name__}: {error}"
        )

        if _is_quota_or_rate_limit_error(
            error
        ):
            return (
                "Gemini's API request quota has "
                "been reached. The document was "
                "retrieved successfully, but LLM-based "
                "answer generation is temporarily "
                "unavailable. Please try again after "
                "the quota resets."
            )

        return (
            "I retrieved relevant document evidence, "
            "but the AI answer-generation service is "
            "currently unavailable. Please try again later."
        )


# ============================================================
# Agent factory
# ============================================================

def create_agent(
    text_store: VectorStore | None = None,
    visual_store: VisualVectorStore | None = None,
) -> ResearchAgent:
    """
    Create and configure the research agent.

    Optional stores are accepted so uploaded documents
    can provide their own isolated vector databases.
    """

    if text_store is None:
        text_store = (
            load_text_vector_store()
        )

    if visual_store is None:
        visual_store = (
            load_visual_vector_store()
        )

    agent = ResearchAgent(
        text_store=text_store,
        visual_store=visual_store,
        query_detector=detect_query_type,
        query_embedder=create_query_embedding,
        text_searcher=search_text,
        visual_searcher=search_visual,
        context_builder=build_context,
        answer_generator=generate_answer,
    )

    return agent


# ============================================================
# CLI entry point
# ============================================================

def main() -> None:
    """
    Simple command-line interface for manual testing.
    """

    print("=" * 60)
    print(
        "Multimodal Agentic Research Assistant"
    )
    print("=" * 60)

    print(
        f"Embedding model: {EMBEDDING_MODEL_NAME}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIMENSION}"
    )

    print(
        f"Gemini model: {GEMINI_MODEL}"
    )

    agent = create_agent()

    print("\nAgent ready.")
    print(
        "Type a question or 'exit' to quit.\n"
    )

    while True:
        query = input(
            "Question: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        if not query:
            continue

        try:
            result = agent.run(
                query
            )

            print("\nAnswer:")
            print(result)

        except Exception as error:
            print(
                "\nAgent error: "
                f"{type(error).__name__}: {error}"
            )

        print()


if __name__ == "__main__":
    main()