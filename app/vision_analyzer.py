import os

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_PATH = "data/images/page_3_image_1.jpeg"


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


def analyze_image(image_path):
    """
    Send an image to Gemini and ask it
    to describe/analyze the visual content.
    """

    # Upload image
    uploaded_file = client.files.upload(
        file=image_path
    )

    prompt = """
You are a research document analysis assistant.

Analyze the provided image carefully.

Determine:

1. What type of visual this is
   (chart, graph, table, diagram, figure, etc.)

2. What the visual is showing.

3. Important labels, values, or text.

4. The main conclusion that can be
   understood from the visual.

5. Any important relationships or trends.

Be factual and do not invent information.
If something is unclear, explicitly say so.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            uploaded_file,
            prompt
        ]
    )

    return response.text


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print(
        f"Analyzing image: {IMAGE_PATH}"
    )

    result = analyze_image(
        IMAGE_PATH
    )

    print("\n====================================")
    print("VISION ANALYSIS")
    print("====================================")

    print(result)