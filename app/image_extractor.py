import os

import pymupdf


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

PDF_PATH = "data/documents/multimodal_research_paper.pdf"

IMAGE_OUTPUT_DIR = "data/images"


# ============================================================
# EXTRACT IMAGES
# ============================================================

def extract_images(
    pdf_path,
    output_dir=IMAGE_OUTPUT_DIR,
):
    """
    Extract all embedded images from a PDF.

    Images are written to the supplied output directory so
    each uploaded document can have its own image workspace.
    """

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    document = pymupdf.open(
        pdf_path
    )

    extracted_images = []

    try:

        for page_number, page in enumerate(
            document
        ):

            images = page.get_images(
                full=True
            )

            print(
                f"Page {page_number + 1}: "
                f"{len(images)} images found"
            )

            for image_number, image in enumerate(
                images,
                start=1
            ):

                xref = image[0]

                image_data = (
                    document.extract_image(
                        xref
                    )
                )

                image_bytes = image_data[
                    "image"
                ]

                image_ext = image_data[
                    "ext"
                ]

                filename = (
                    f"page_{page_number + 1}"
                    f"_image_{image_number}"
                    f".{image_ext}"
                )

                image_path = os.path.join(
                    output_dir,
                    filename
                )

                with open(
                    image_path,
                    "wb"
                ) as file:

                    file.write(
                        image_bytes
                    )

                extracted_images.append(
                    {
                        "page":
                            page_number + 1,

                        "image_number":
                            image_number,

                        "path":
                            image_path,

                        "filename":
                            filename,
                    }
                )

                print(
                    f"  Saved: {image_path}"
                )

    finally:

        document.close()

    return extracted_images


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    images = extract_images(
        PDF_PATH,
        IMAGE_OUTPUT_DIR
    )

    print(
        f"\nTotal images extracted: "
        f"{len(images)}"
    )