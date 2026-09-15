import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

METADATA_FILE = (
    "data/visual_analysis/"
    "visual_metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"

EXPECTED_VISUAL_COUNT = 17


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

print(
    "Loading embedding model..."
)

model = SentenceTransformer(
    MODEL_NAME
)


# ==================================================
# LOAD VISUAL METADATA
# ==================================================

def load_visual_metadata():

    if not os.path.exists(
        METADATA_FILE
    ):
        raise FileNotFoundError(
            f"Could not find: {METADATA_FILE}"
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    if not isinstance(
        metadata,
        list
    ):
        raise ValueError(
            "visual_metadata.json must "
            "contain a list."
        )

    print(
        f"Visual documents found: "
        f"{len(metadata)}"
    )

    # ----------------------------------------------
    # Validate visual records
    # ----------------------------------------------

    valid_metadata = []

    for index, item in enumerate(
        metadata,
        start=1
    ):

        if not isinstance(
            item,
            dict
        ):
            print(
                f"Warning: visual record "
                f"{index} is invalid."
            )

            continue

        if not item.get(
            "analysis"
        ):
            print(
                f"Warning: visual record "
                f"{index} has no analysis."
            )

        valid_metadata.append(
            item
        )

    metadata = valid_metadata

    print(
        f"Valid visual documents: "
        f"{len(metadata)}"
    )

    return metadata


# ==================================================
# BUILD SEARCH TEXT
# ==================================================

def build_search_text(
    item
):
    """
    Creates a rich searchable representation
    of each visual.
    """

    source = item.get(
        "source",
        ""
    )

    page = item.get(
        "page",
        ""
    )

    image = item.get(
        "image",
        ""
    )

    analysis = item.get(
        "analysis",
        ""
    )

    return (
        f"Source: {source}. "
        f"Page: {page}. "
        f"Image: {image}. "
        f"Visual analysis: {analysis}"
    )


# ==================================================
# TOKENIZE TEXT
# ==================================================

def tokenize(
    text
):
    """
    Convert text into normalized tokens.
    """

    return set(
        re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )
    )


# ==================================================
# BUILD VISUAL EMBEDDINGS
# ==================================================

def build_visual_embeddings(
    metadata
):

    texts = []

    for item in metadata:

        texts.append(
            build_search_text(
                item
            )
        )

    if not texts:
        raise ValueError(
            "No visual metadata found."
        )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    return embeddings


# ==================================================
# HYBRID VISUAL SEARCH
# ==================================================

def search_visuals(
    query,
    metadata,
    embeddings,
    top_k=3
):

    # ------------------------------------------------
    # Query embedding
    # ------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # ------------------------------------------------
    # Semantic similarity
    # ------------------------------------------------

    semantic_scores = (
        embeddings
        @ query_embedding
    )

    # ------------------------------------------------
    # Keyword matching
    # ------------------------------------------------

    query_words = tokenize(
        query
    )

    keyword_scores = []

    for item in metadata:

        search_text = (
            build_search_text(
                item
            )
        )

        search_words = tokenize(
            search_text
        )

        if not query_words:
            score = 0.0

        else:

            matches = (
                query_words
                &
                search_words
            )

            score = (
                len(matches)
                /
                len(query_words)
            )

        keyword_scores.append(
            score
        )

    keyword_scores = np.asarray(
        keyword_scores,
        dtype="float32"
    )

    # ------------------------------------------------
    # Hybrid score
    # ------------------------------------------------

    final_scores = (
        0.75 * semantic_scores
        +
        0.25 * keyword_scores
    )

    # ------------------------------------------------
    # Rank
    # ------------------------------------------------

    top_k = min(
        top_k,
        len(metadata)
    )

    top_indices = np.argsort(
        final_scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        item = metadata[
            index
        ]

        results.append(
            {
                "source":
                    item.get(
                        "source",
                        "Unknown"
                    ),

                "page":
                    item.get(
                        "page",
                        "Unknown"
                    ),

                "image":
                    item.get(
                        "image",
                        "Unknown"
                    ),

                "path":
                    item.get(
                        "path",
                        ""
                    ),

                "score":
                    float(
                        final_scores[index]
                    ),

                "semantic_score":
                    float(
                        semantic_scores[index]
                    ),

                "keyword_score":
                    float(
                        keyword_scores[index]
                    ),

                "analysis":
                    item.get(
                        "analysis",
                        ""
                    )
            }
        )

    return results


# ==================================================
# DISPLAY RESULTS
# ==================================================

def display_results(
    results
):

    print(
        "\n===================================="
    )

    print(
        "VISUAL SEARCH RESULTS"
    )

    print(
        "===================================="
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {number}"
        )

        print(
            "------------------------------------"
        )

        print(
            f"Source : "
            f"{result['source']}"
        )

        print(
            f"Page   : "
            f"{result['page']}"
        )

        print(
            f"Image  : "
            f"{result['image']}"
        )

        print(
            f"Hybrid Score   : "
            f"{result['score']:.4f}"
        )

        print(
            f"Semantic Score : "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Keyword Score  : "
            f"{result['keyword_score']:.4f}"
        )

        print(
            "\nAnalysis:"
        )

        print(
            result["analysis"]
        )


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    # ------------------------------------------------
    # Load metadata
    # ------------------------------------------------

    metadata = (
        load_visual_metadata()
    )

    # ------------------------------------------------
    # Check expected count
    # ------------------------------------------------

    if len(metadata) != (
        EXPECTED_VISUAL_COUNT
    ):

        print(
            "\nWARNING:"
        )

        print(
            f"Expected "
            f"{EXPECTED_VISUAL_COUNT} "
            f"visuals, but found "
            f"{len(metadata)}."
        )

        print(
            "The visual metadata needs "
            "to be regenerated."
        )

    # ------------------------------------------------
    # Build embeddings
    # ------------------------------------------------

    embeddings = (
        build_visual_embeddings(
            metadata
        )
    )

    print(
        f"\nEmbedding shape: "
        f"{embeddings.shape}"
    )

    # ------------------------------------------------
    # Validate embedding count
    # ------------------------------------------------

    if len(metadata) != (
        embeddings.shape[0]
    ):

        raise ValueError(
            "Metadata count does not match "
            "embedding count."
        )

    # ------------------------------------------------
    # Interactive search
    # ------------------------------------------------

    while True:

        query = input(
            "\nSearch visual knowledge "
            "(type 'exit' to quit): "
        )

        if (
            query
            .lower()
            .strip()
            == "exit"
        ):

            print(
                "\nExiting visual search."
            )

            break

        if not query.strip():

            print(
                "Please enter a question."
            )

            continue

        # ------------------------------------------------
        # Search
        # ------------------------------------------------

        results = search_visuals(
            query,
            metadata,
            embeddings,
            top_k=3
        )

        # ------------------------------------------------
        # Display
        # ------------------------------------------------

        display_results(
            results
        )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    main()