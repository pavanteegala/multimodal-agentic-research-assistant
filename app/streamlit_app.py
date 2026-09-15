# ============================================================
# PROJECT ROOT / PYTHON PATH FIX
# ============================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import contextlib
import hashlib
import io
import re

import pymupdf
import streamlit as st

from app.multimodal_rag import create_agent
from app.uploaded_document import process_document


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal Research Assistant",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-top: 1rem;
    }

    .source-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f7f7f7;
        margin-top: 1rem;
    }

    .evidence-box {
        padding: 0.5rem 0;
        margin-top: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "agent" not in st.session_state:
    st.session_state.agent = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_root" not in st.session_state:
    st.session_state.document_root = None

if "document_pdf" not in st.session_state:
    st.session_state.document_pdf = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# CITATION HELPERS
# ============================================================

def citation_label(citation):
    """
    Convert a structured citation dictionary into a readable
    source label for the UI.
    """

    if isinstance(citation, str):
        return citation

    if not isinstance(citation, dict):
        return str(citation)

    citation_type = citation.get("type", "text")

    page = citation.get(
        "page",
        "unknown",
    )

    image = citation.get(
        "image"
    )

    chunk_id = citation.get(
        "chunk_id"
    )

    source = citation.get(
        "source",
        "unknown source",
    )

    if citation_type == "visual":

        if image:
            return (
                f"{source} — Page {page} — "
                f"{image}"
            )

        return (
            f"{source} — Page {page} — "
            "Visual evidence"
        )

    if chunk_id is not None:

        return (
            f"{source} — Page {page} — "
            f"Chunk {chunk_id}"
        )

    return (
        f"{source} — Page {page}"
    )


def is_visual_citation(citation):
    """
    Determine whether a citation refers to visual evidence.
    """

    if isinstance(citation, dict):
        if citation.get("type") == "visual":
            return True

        if citation.get("image"):
            return True

    reference = citation_label(
        citation
    )

    return extract_image_name(
        reference
    ) is not None


# ============================================================
# HELPER — EXTRACT IMAGE NAME
# ============================================================

def extract_image_name(reference):
    """
    Extract an image filename from a source reference.

    Supports examples such as:
        page_1_image_1.png
        page_2_image_1.jpeg
        page_3_image_2.webp
    """

    reference = str(reference)

    match = re.search(
        r"(page_\d+_image_\d+\.(?:png|jpg|jpeg|webp))",
        reference,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# HELPER — EXTRACT PAGE NUMBER
# ============================================================

def extract_page_number(reference):
    """
    Extract page number from a source reference.
    """

    reference = str(reference)

    match = re.search(
        r"Page\s+(\d+)",
        reference,
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# HELPER — GET IMAGE PATH
# ============================================================

def get_image_path(citation):
    """
    Locate an extracted visual belonging to the active document.
    """

    # --------------------------------------------------------
    # Structured citation
    # --------------------------------------------------------

    if isinstance(citation, dict):

        image_name = citation.get(
            "image"
        )

        metadata = citation.get(
            "metadata",
            {},
        )

        if not image_name and isinstance(
            metadata,
            dict,
        ):
            image_name = metadata.get(
                "image",
                metadata.get(
                    "filename"
                ),
            )

    else:

        image_name = extract_image_name(
            citation
        )

    if not image_name:
        return None

    # --------------------------------------------------------
    # Uploaded document
    # --------------------------------------------------------

    if st.session_state.document_root:

        document_root = Path(
            st.session_state.document_root
        )

        possible_paths = [
            document_root / "images" / image_name,
            document_root / image_name,
        ]

        for path in possible_paths:

            if path.exists():
                return path

    # --------------------------------------------------------
    # Default document
    # --------------------------------------------------------

    default_image = (
        PROJECT_ROOT
        / "data"
        / "images"
        / image_name
    )

    if default_image.exists():
        return default_image

    return None


# ============================================================
# HELPER — GET PDF PATH
# ============================================================

def get_pdf_path():

    # Uploaded document
    if st.session_state.document_pdf:

        path = Path(
            st.session_state.document_pdf
        )

        if path.exists():
            return path

    # Default document
    default_pdf = (
        PROJECT_ROOT
        / "data"
        / "documents"
        / "multimodal_research_paper.pdf"
    )

    if default_pdf.exists():
        return default_pdf

    return None


# ============================================================
# HELPER — RENDER PDF PAGE
# ============================================================

def render_pdf_page(
    page_number,
):
    """
    Render a PDF page into a PNG image for browser display.
    """

    pdf_path = get_pdf_path()

    if pdf_path is None:
        return None

    try:

        document = pymupdf.open(
            str(pdf_path)
        )

        page_index = page_number - 1

        if (
            page_index < 0
            or page_index >= len(document)
        ):

            document.close()
            return None

        page = document[
            page_index
        ]

        matrix = pymupdf.Matrix(
            1.5,
            1.5,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image_bytes = pixmap.tobytes(
            "png"
        )

        document.close()

        return image_bytes

    except Exception:
        return None


# ============================================================
# HELPER — DISPLAY VISUAL EVIDENCE
# ============================================================

def display_visual_sources(
    citations,
):
    """
    Display actual extracted visual evidence.
    """

    visual_items = []

    for citation in citations:

        if not is_visual_citation(
            citation
        ):
            continue

        image_path = get_image_path(
            citation
        )

        if image_path:

            visual_items.append(
                (
                    citation,
                    image_path,
                )
            )

    if not visual_items:
        return

    with st.expander(
        "View visual evidence",
        expanded=True,
    ):

        for citation, image_path in visual_items:

            st.caption(
                citation_label(
                    citation
                )
            )

            st.image(
                str(image_path),
                use_container_width=True,
            )


# ============================================================
# HELPER — DISPLAY PDF PAGE EVIDENCE
# ============================================================

def display_pdf_sources(
    citations,
):
    """
    Display the actual PDF page associated with
    text-based source citations.
    """

    page_items = []

    for citation in citations:

        # Visual evidence already has its own viewer.
        if is_visual_citation(
            citation
        ):
            continue

        reference = citation_label(
            citation
        )

        page_number = (
            extract_page_number(
                reference
            )
        )

        if page_number:

            page_items.append(
                (
                    citation,
                    page_number,
                )
            )

    if not page_items:
        return

    unique_pages = []

    seen_pages = set()

    for citation, page_number in page_items:

        if page_number in seen_pages:
            continue

        seen_pages.add(
            page_number
        )

        unique_pages.append(
            (
                citation,
                page_number,
            )
        )

    with st.expander(
        "View source page",
        expanded=False,
    ):

        for citation, page_number in unique_pages:

            page_image = render_pdf_page(
                page_number
            )

            st.caption(
                f"Page {page_number} — "
                f"{citation_label(citation)}"
            )

            if page_image:

                st.image(
                    page_image,
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Unable to render this PDF page."
                )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    "🔎 Multimodal Agentic Research Assistant"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Upload a PDF and ask questions using text,
    tables, charts, figures, and other visual evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Document"
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help=(
            "Upload a research paper, report, "
            "manual, or other PDF."
        ),
    )

    st.divider()

    # ========================================================
    # UPLOADED DOCUMENT
    # ========================================================

    if uploaded_file is not None:

        file_bytes = (
            uploaded_file.getvalue()
        )

        document_id = (
            hashlib.sha256(
                file_bytes
            ).hexdigest()[:16]
        )

        if (
            st.session_state.document_id
            != document_id
        ):

            st.session_state.agent = None
            st.session_state.messages = []

            with st.status(
                "Processing document...",
                expanded=True,
            ) as status:

                try:

                    st.write(
                        "Saving PDF..."
                    )

                    result = process_document(
                        file_bytes,
                        uploaded_file.name,
                    )

                    st.write(
                        "Text index ready."
                    )

                    st.write(
                        "Visual index ready."
                    )

                    buffer = io.StringIO()

                    with contextlib.redirect_stdout(
                        buffer
                    ):

                        agent = create_agent(
                            text_store=result[
                                "text_store"
                            ],
                            visual_store=result[
                                "visual_store"
                            ],
                        )

                    st.session_state.agent = agent

                    st.session_state.document_id = (
                        result[
                            "document_id"
                        ]
                    )

                    st.session_state.document_name = (
                        result[
                            "filename"
                        ]
                    )

                    st.session_state.document_pdf = (
                        result[
                            "pdf_path"
                        ]
                    )

                    # IMPORTANT:
                    #
                    # document.pdf is stored inside:
                    #
                    # data/processed/<document_id>/
                    #
                    # images are stored inside:
                    #
                    # data/processed/<document_id>/images/
                    #
                    # Therefore document_root should point
                    # to the document workspace itself.

                    st.session_state.document_root = str(
                        Path(
                            result[
                                "pdf_path"
                            ]
                        ).parent
                    )

                    status.update(
                        label="Document ready",
                        state="complete",
                        expanded=False,
                    )

                except Exception as error:

                    status.update(
                        label="Document processing failed",
                        state="error",
                        expanded=True,
                    )

                    st.error(
                        str(error)
                    )

        if st.session_state.document_name:

            st.success(
                "Document ready"
            )

    # ========================================================
    # DEFAULT DOCUMENT
    # ========================================================

    if uploaded_file is None:

        if (
            st.session_state.agent is None
        ):

            with st.spinner(
                "Loading default research paper..."
            ):

                try:

                    buffer = io.StringIO()

                    with contextlib.redirect_stdout(
                        buffer
                    ):

                        st.session_state.agent = (
                            create_agent()
                        )

                    st.session_state.document_name = (
                        "multimodal_research_paper.pdf"
                    )

                    st.session_state.document_id = (
                        "default"
                    )

                    # Default document images live in:
                    #
                    # data/images/

                    st.session_state.document_root = (
                        str(
                            PROJECT_ROOT
                            / "data"
                        )
                    )

                    st.session_state.document_pdf = str(
                        PROJECT_ROOT
                        / "data"
                        / "documents"
                        / "multimodal_research_paper.pdf"
                    )

                except Exception as error:

                    st.error(
                        "Unable to load the default document."
                    )

                    st.exception(
                        error
                    )

                    st.stop()

        st.success(
            "Default research paper loaded"
        )

    # ========================================================
    # CAPABILITIES
    # ========================================================

    st.divider()

    st.header(
        "Capabilities"
    )

    st.write(
        "• Text retrieval"
    )

    st.write(
        "• Table and figure understanding"
    )

    st.write(
        "• Visual retrieval"
    )

    st.write(
        "• Agentic task classification"
    )

    st.write(
        "• Multi-hop research"
    )

    st.write(
        "• Source-backed answers"
    )

    st.write(
        "• Visual evidence display"
    )

    st.write(
        "• PDF page preview"
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# ACTIVE DOCUMENT
# ============================================================

if st.session_state.document_name:

    st.info(
        f"Active document: "
        f"{st.session_state.document_name}"
    )


# ============================================================
# PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        citations = message.get(
            "citations",
            [],
        )

        if (
            message["role"] == "assistant"
            and citations
        ):

            st.markdown(
                "**Sources**"
            )

            for citation in citations:

                st.markdown(
                    f"- {citation_label(citation)}"
                )

            display_visual_sources(
                citations
            )

            display_pdf_sources(
                citations
            )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the document..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Store user question
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Researching the document..."
        ):

            try:

                buffer = io.StringIO()

                with contextlib.redirect_stdout(
                    buffer
                ):

                    result = (
                        st.session_state.agent.run(
                            question
                        )
                    )

                answer = result.get(
                    "answer",
                    "No answer was generated.",
                )

                # IMPORTANT:
                #
                # agent.py returns `citations`,
                # not `references`.

                citations = result.get(
                    "citations",
                    [],
                )

                # ------------------------------------------------
                # ANSWER
                # ------------------------------------------------

                st.markdown(
                    '<div class="answer-box">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    answer
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                if citations:

                    st.markdown(
                        '<div class="source-box">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "**Sources**"
                    )

                    for citation in citations:

                        st.markdown(
                            f"- {citation_label(citation)}"
                        )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # --------------------------------------------
                    # Actual visual evidence
                    # --------------------------------------------

                    display_visual_sources(
                        citations
                    )

                    # --------------------------------------------
                    # Actual text PDF pages
                    # --------------------------------------------

                    display_pdf_sources(
                        citations
                    )

                # ------------------------------------------------
                # Save assistant response
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )

            except Exception as error:

                st.error(
                    "Unable to answer the question."
                )

                st.exception(
                    error
                )