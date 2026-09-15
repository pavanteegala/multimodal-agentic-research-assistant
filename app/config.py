from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"

VECTOR_STORE_DIR = DATA_DIR / "vector_store"
VISUAL_VECTOR_STORE_DIR = DATA_DIR / "visual_vector_store"

VISUAL_ANALYSIS_DIR = DATA_DIR / "visual_analysis"


# ============================================================
# Default document
# ============================================================

DEFAULT_PDF_PATH = DOCUMENTS_DIR / "multimodal_research_paper.pdf"


# ============================================================
# Embedding configuration
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


# ============================================================
# Gemini configuration
# ============================================================

GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# Retrieval configuration
# ============================================================

TEXT_TOP_K = 3
VISUAL_TOP_K = 3


# ============================================================
# Application limits
# ============================================================

MAX_TEXT_SOURCES = 3
MAX_VISUAL_SOURCES = 3


def ensure_directories() -> None:
    """
    Create the main project directories if they do not exist.
    """

    directories = [
        DATA_DIR,
        DOCUMENTS_DIR,
        IMAGES_DIR,
        PROCESSED_DIR,
        VECTOR_STORE_DIR,
        VISUAL_VECTOR_STORE_DIR,
        VISUAL_ANALYSIS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)