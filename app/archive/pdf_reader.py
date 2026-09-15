import pymupdf


# Path to our PDF
pdf_path = "data/documents/research_paper.pdf"


# Open the PDF
document = pymupdf.open(pdf_path)


# Show number of pages
print(f"Number of pages: {len(document)}")


# Read every page
for page_number, page in enumerate(document):

    # Extract text from the page
    text = page.get_text()

    print(f"\n--- PAGE {page_number + 1} ---\n")

    print(text[:1000])