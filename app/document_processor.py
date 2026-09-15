import os
import pymupdf

from app.chunker import create_chunks


PDF_PATH = "data/documents/multimodal_research_paper.pdf"


def process_pdf(pdf_path):
    """
    Read a PDF and convert it into
    chunks with metadata.
    """

    document = pymupdf.open(pdf_path)

    all_chunks = []

    chunk_id = 0

    # Get the actual PDF filename
    source_name = os.path.basename(pdf_path)

    for page_number, page in enumerate(document):

        # ------------------------------------------
        # Extract text from page
        # ------------------------------------------

        text = page.get_text().strip()

        # Skip empty pages
        if not text:
            continue

        # ------------------------------------------
        # Create chunks
        # ------------------------------------------

        chunks = create_chunks(
            text,
            chunk_size=500,
            overlap=80
        )

        # ------------------------------------------
        # Add metadata
        # ------------------------------------------

        for chunk in chunks:

            chunk_id += 1

            chunk_data = {
                "chunk_id": chunk_id,
                "text": chunk,
                "source": source_name,
                "page": page_number + 1
            }

            all_chunks.append(chunk_data)

    document.close()

    return all_chunks


if __name__ == "__main__":

    chunks = process_pdf(PDF_PATH)

    print(
        f"Total chunks created: {len(chunks)}"
    )

    for chunk in chunks:

        print("\n-----------------------------")

        print(
            f"Chunk ID : {chunk['chunk_id']}"
        )

        print(
            f"Source  : {chunk['source']}"
        )

        print(
            f"Page    : {chunk['page']}"
        )

        print("-----------------------------")

        print(
            chunk["text"][:300]
        )