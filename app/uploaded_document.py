import hashlib
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.document_processor import process_pdf
from app.embedder import create_embeddings
from app.image_extractor import extract_images
from app.vision_analyzer import analyze_image
from app.vector_store import VectorStore
from app.visual_vector_store import VisualVectorStore


# ============================================================
# CONFIGURATION
# ============================================================

BASE_PROCESSED_DIR = Path(
    "data/processed"
)

EMBEDDING_DIMENSION = 384

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)


# ============================================================
# DOCUMENT ID
# ============================================================

def create_document_id(
    file_bytes
):
    return hashlib.sha256(
        file_bytes
    ).hexdigest()[:16]


# ============================================================
# DOCUMENT PATHS
# ============================================================

def get_document_paths(
    document_id
):

    root = (
        BASE_PROCESSED_DIR
        / document_id
    )

    return {
        "root":
            root,

        "pdf":
            root / "document.pdf",

        "images":
            root / "images",

        "visual_analysis":
            root / "visual_analysis",

        "visual_metadata":
            root
            / "visual_analysis"
            / "visual_metadata.json",

        "text_store":
            root / "vector_store",

        "visual_store":
            root / "visual_vector_store",

        "manifest":
            root / "manifest.json",
    }


# ============================================================
# SAVE PDF
# ============================================================

def save_uploaded_pdf(
    file_bytes,
    original_filename,
):

    document_id = create_document_id(
        file_bytes
    )

    paths = get_document_paths(
        document_id
    )

    paths["root"].mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        paths["pdf"],
        "wb"
    ) as file:

        file.write(
            file_bytes
        )

    manifest = {
        "document_id":
            document_id,

        "original_filename":
            original_filename,

        "pdf_path":
            str(paths["pdf"]),
    }

    with open(
        paths["manifest"],
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2
        )

    return (
        document_id,
        paths
    )


# ============================================================
# CHUNK → TEXT
# ============================================================

def get_chunk_text(
    chunk
):

    if isinstance(
        chunk,
        str
    ):
        return chunk

    if isinstance(
        chunk,
        dict
    ):

        return str(
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("page_content")
            or ""
        )

    return str(
        chunk
    )


# ============================================================
# GET PAGE NUMBER
# ============================================================

def get_page_number(
    filename
):

    match = re.search(
        r"page_(\d+)",
        filename,
        re.IGNORECASE
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# BUILD TEXT VECTOR STORE
# ============================================================

def build_text_store(
    pdf_path,
    text_store_path,
    source_filename,
):

    index_path = (
        text_store_path
        / "index.faiss"
    )

    metadata_path = (
        text_store_path
        / "metadata.json"
    )

    # --------------------------------------------------------
    # Reuse completed text index
    # --------------------------------------------------------

    if (
        index_path.exists()
        and metadata_path.exists()
    ):

        store = VectorStore(
            dimension=EMBEDDING_DIMENSION
        )

        store.load(
            str(text_store_path)
        )

        return store

    print(
        "Extracting text and creating chunks..."
    )

    chunks = process_pdf(
        str(pdf_path)
    )

    if not chunks:

        raise ValueError(
            "No text could be extracted from the PDF."
        )

    metadata = []

    for index, chunk in enumerate(
        chunks
    ):

        if isinstance(
            chunk,
            dict
        ):

            item = dict(
                chunk
            )

        else:

            item = {
                "text":
                    str(chunk)
            }

        item["source"] = (
            source_filename
        )

        if "chunk_id" not in item:

            item["chunk_id"] = (
                index + 1
            )

        metadata.append(
            item
        )

    # --------------------------------------------------------
    # ONLY strings go into SentenceTransformer
    # --------------------------------------------------------

    texts = [
        get_chunk_text(
            item
        )
        for item in metadata
    ]

    valid = [
        (
            text,
            item
        )
        for text, item in zip(
            texts,
            metadata
        )
        if text.strip()
    ]

    if not valid:

        raise ValueError(
            "No usable text chunks were found."
        )

    texts = [
        pair[0]
        for pair in valid
    ]

    metadata = [
        pair[1]
        for pair in valid
    ]

    print(
        f"Text chunks: {len(texts)}"
    )

    embeddings = create_embeddings(
        texts
    )

    store = VectorStore(
        dimension=EMBEDDING_DIMENSION
    )

    store.add_embeddings(
        embeddings,
        metadata
    )

    store.save(
        str(text_store_path)
    )

    return store


# ============================================================
# BUILD VISUAL VECTOR STORE
# ============================================================

def build_visual_store(
    pdf_path,
    images_dir,
    visual_analysis_dir,
    visual_store_path,
    source_filename,
):

    visual_metadata_path = (
        visual_analysis_dir
        / "visual_metadata.json"
    )

    index_path = (
        visual_store_path
        / "index.faiss"
    )

    metadata_path = (
        visual_store_path
        / "metadata.json"
    )

    # --------------------------------------------------------
    # Reuse ONLY if the store actually contains vectors
    # --------------------------------------------------------

    if (
        index_path.exists()
        and metadata_path.exists()
    ):

        store = VisualVectorStore(
            dimension=EMBEDDING_DIMENSION
        )

        store.load(
            str(visual_store_path)
        )

        if store.index.ntotal > 0:

            return store

        print(
            "Existing visual store is empty. "
            "Rebuilding visual index..."
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    images_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    visual_analysis_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    visual_store_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Extract images
    # --------------------------------------------------------

    image_files = [
        file
        for file in images_dir.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }
    ]

    if not image_files:

        print(
            "Extracting images from PDF..."
        )

        extract_images(
            str(pdf_path),
            str(images_dir)
        )

        image_files = [
            file
            for file in images_dir.iterdir()
            if file.is_file()
            and file.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }
        ]

    image_files.sort(
        key=lambda file: file.name
    )

    print(
        f"Visual images found: "
        f"{len(image_files)}"
    )

    # --------------------------------------------------------
    # Load cached analysis
    # --------------------------------------------------------

    cached_metadata = []

    if visual_metadata_path.exists():

        try:

            with open(
                visual_metadata_path,
                "r",
                encoding="utf-8"
            ) as file:

                cached_metadata = (
                    json.load(file)
                )

        except Exception:

            cached_metadata = []

    cached_by_image = {
        item.get("image"):
            item
        for item in cached_metadata
        if item.get("image")
    }

    final_metadata = []

    # --------------------------------------------------------
    # Analyze every visual
    # --------------------------------------------------------

    for number, image_path in enumerate(
        image_files,
        start=1
    ):

        filename = image_path.name

        print(
            f"Visual {number}/{len(image_files)}: "
            f"{filename}"
        )

        # ----------------------------------------------------
        # Reuse analysis
        # ----------------------------------------------------

        if filename in cached_by_image:

            cached = dict(
                cached_by_image[
                    filename
                ]
            )

            cached["source"] = (
                source_filename
            )

            final_metadata.append(
                cached
            )

            print(
                "  Using cached visual analysis."
            )

            continue

        # ----------------------------------------------------
        # Gemini visual analysis
        # ----------------------------------------------------

        try:

            print(
                "  Sending image to Gemini..."
            )

            analysis = analyze_image(
                str(image_path)
            )

            if not analysis:

                raise ValueError(
                    "Gemini returned empty visual analysis."
                )

            item = {
                "source":
                    source_filename,

                "page":
                    get_page_number(
                        filename
                    ),

                "image":
                    filename,

                "path":
                    str(image_path),

                "analysis":
                    analysis,
            }

            final_metadata.append(
                item
            )

            print(
                "  Visual analysis completed."
            )

            # Save immediately
            with open(
                visual_metadata_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    final_metadata,
                    file,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception as error:

            print(
                f"  Visual analysis failed: "
                f"{error}"
            )

    # --------------------------------------------------------
    # Require successful visual analysis
    # --------------------------------------------------------

    if image_files and not final_metadata:

        raise RuntimeError(
            "The PDF contains images, but none of the "
            "images could be analyzed successfully."
        )

    # --------------------------------------------------------
    # Save visual metadata
    # --------------------------------------------------------

    with open(
        visual_metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # No images
    # --------------------------------------------------------

    if not final_metadata:

        print(
            "No visual content found."
        )

        store = VisualVectorStore(
            dimension=EMBEDDING_DIMENSION
        )

        store.save(
            str(visual_store_path)
        )

        return store

    # --------------------------------------------------------
    # Create visual embeddings
    # --------------------------------------------------------

    print(
        "Creating visual embeddings..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    searchable_text = []

    for item in final_metadata:

        searchable_text.append(
            f"""
Source: {item.get('source', '')}
Page: {item.get('page', '')}
Image: {item.get('image', '')}

Visual analysis:
{item.get('analysis', '')}
"""
        )

    embeddings = model.encode(
        searchable_text,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    print(
        f"Visual embedding shape: "
        f"{embeddings.shape}"
    )

    store = VisualVectorStore(
        dimension=embeddings.shape[1]
    )

    store.add_embeddings(
        embeddings,
        final_metadata
    )

    store.save(
        str(visual_store_path)
    )

    print(
        f"Visual vectors created: "
        f"{store.index.ntotal}"
    )

    return store


# ============================================================
# PROCESS COMPLETE DOCUMENT
# ============================================================

def process_document(
    file_bytes,
    original_filename,
):

    print(
        f"Processing document: "
        f"{original_filename}"
    )

    (
        document_id,
        paths,
    ) = save_uploaded_pdf(
        file_bytes,
        original_filename,
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_store = build_text_store(
        paths["pdf"],
        paths["text_store"],
        original_filename,
    )

    # --------------------------------------------------------
    # VISUAL
    # --------------------------------------------------------

    visual_store = build_visual_store(
        paths["pdf"],
        paths["images"],
        paths["visual_analysis"],
        paths["visual_store"],
        original_filename,
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    image_count = len(
        [
            file
            for file in paths["images"].iterdir()
            if file.is_file()
            and file.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
            }
        ]
    )

    visual_vector_count = (
        visual_store.index.ntotal
    )

    print(
        "\nDocument processing complete."
    )

    print(
        f"Text chunks: "
        f"{text_store.index.ntotal}"
    )

    print(
        f"Images: "
        f"{image_count}"
    )

    print(
        f"Visual vectors: "
        f"{visual_vector_count}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if image_count > 0 and visual_vector_count == 0:

        raise RuntimeError(
            "The document contains images but the visual "
            "vector store contains zero vectors."
        )

    return {
        "document_id":
            document_id,

        "filename":
            original_filename,

        "pdf_path":
            str(paths["pdf"]),

        "text_store":
            text_store,

        "visual_store":
            visual_store,

        "text_store_path":
            str(paths["text_store"]),

        "visual_store_path":
            str(paths["visual_store"]),

        "image_count":
            image_count,

        "visual_vector_count":
            visual_vector_count,
    }