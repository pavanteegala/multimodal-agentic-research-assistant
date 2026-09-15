import json
import os

from dotenv import load_dotenv
from google import genai


# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_DIRECTORY = "data/images"

OUTPUT_DIRECTORY = "data/visual_analysis"

OUTPUT_FILE = (
    "data/visual_analysis/"
    "visual_metadata.json"
)


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found "
        "in the .env file"
    )


# ==================================================
# GEMINI CLIENT
# ==================================================

client = genai.Client(
    api_key=api_key
)


# ==================================================
# ANALYZE ONE IMAGE
# ==================================================

def analyze_image(image_path):

    print(
        f"Uploading image: {image_path}"
    )

    uploaded_file = client.files.upload(
        file=image_path
    )

    prompt = """
You are analyzing a visual extracted from
a multimodal research paper.

Analyze the image carefully.

Return the following:

1. Visual type
2. Title or caption if visible
3. Important text and labels
4. Important numerical values
5. Main findings or relationships
6. Important trends or comparisons

Pay special attention to:

- tables
- figures
- charts
- graphs
- diagrams
- column and row labels
- numerical values
- captions
- relationships between values

For tables, identify the complete table structure
and the important values.

For charts and graphs, identify axes, labels,
legends, values, and important trends.

Do not invent information.

If something cannot be read clearly,
say so explicitly.

Return the result as plain text.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            uploaded_file,
            prompt
        ]
    )

    return response.text


# ==================================================
# LOAD EXISTING RESULTS
# ==================================================

def load_existing_results():

    if not os.path.exists(
        OUTPUT_FILE
    ):
        return []

    try:

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        print(
            f"Could not read existing metadata: "
            f"{error}"
        )

        return []


# ==================================================
# SAVE RESULTS
# ==================================================

def save_results(results):

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )


# ==================================================
# EXTRACT PAGE NUMBER
# ==================================================

def get_page_number(filename):

    try:

        # Example:
        # page_17_image_1.png

        parts = filename.split("_")

        return int(
            parts[1]
        )

    except Exception:

        return None


# ==================================================
# FIND ALL IMAGES
# ==================================================

def get_image_files():

    if not os.path.exists(
        IMAGE_DIRECTORY
    ):
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{IMAGE_DIRECTORY}"
        )

    image_files = []

    for filename in os.listdir(
        IMAGE_DIRECTORY
    ):

        if filename.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp"
            )
        ):

            image_files.append(
                filename
            )

    image_files.sort(
        key=lambda filename: (
            get_page_number(filename) or 999,
            filename
        )
    )

    return image_files


# ==================================================
# ANALYZE ALL IMAGES
# ==================================================

def analyze_all_images():

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    # ------------------------------------------------
    # Load existing metadata
    # ------------------------------------------------

    results = load_existing_results()

    existing_by_image = {
        item["image"]: item
        for item in results
        if "image" in item
    }

    print(
        f"Existing metadata records: "
        f"{len(existing_by_image)}"
    )

    # ------------------------------------------------
    # Find all actual images
    # ------------------------------------------------

    image_files = get_image_files()

    print(
        f"Actual images found: "
        f"{len(image_files)}"
    )

    # ------------------------------------------------
    # Display images
    # ------------------------------------------------

    print(
        "\nImages discovered:"
    )

    for number, filename in enumerate(
        image_files,
        start=1
    ):

        print(
            f"  {number:02d}. "
            f"{filename}"
        )

    # ------------------------------------------------
    # Process images
    # ------------------------------------------------

    final_results = []

    for number, filename in enumerate(
        image_files,
        start=1
    ):

        # --------------------------------------------
        # Reuse existing analysis if available
        # --------------------------------------------

        if filename in existing_by_image:

            print(
                f"\n[{number}/{len(image_files)}] "
                f"Already analyzed: {filename}"
            )

            final_results.append(
                existing_by_image[filename]
            )

            continue

        # --------------------------------------------
        # Analyze new image
        # --------------------------------------------

        image_path = os.path.join(
            IMAGE_DIRECTORY,
            filename
        )

        print(
            f"\n[{number}/{len(image_files)}] "
            f"Analyzing {filename}..."
        )

        try:

            analysis = analyze_image(
                image_path
            )

            result = {
                "source":
                    "multimodal_research_paper.pdf",

                "page":
                    get_page_number(
                        filename
                    ),

                "image":
                    filename,

                "path":
                    image_path,

                "analysis":
                    analysis
            }

            final_results.append(
                result
            )

            # ----------------------------------------
            # Save immediately
            # ----------------------------------------

            save_results(
                final_results
            )

            print(
                "Analysis completed and saved."
            )

        except Exception as error:

            print(
                f"ERROR analyzing "
                f"{filename}: {error}"
            )

            print(
                "Skipping this image."
            )

    # ------------------------------------------------
    # Final save
    # ------------------------------------------------

    save_results(
        final_results
    )

    # ------------------------------------------------
    # Final summary
    # ------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        "VISUAL PROCESSING COMPLETE"
    )

    print(
        "===================================="
    )

    print(
        f"Images found       : "
        f"{len(image_files)}"
    )

    print(
        f"Metadata records   : "
        f"{len(final_results)}"
    )

    print(
        f"Metadata file      : "
        f"{OUTPUT_FILE}"
    )

    print(
        "===================================="
    )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    analyze_all_images()