import os

from dotenv import load_dotenv
from google import genai


# Load variables from .env
load_dotenv()

# Get the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Check that the API key exists
if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Send a request to Gemini
response = client.interactions.create(
    model="gemini-3.6-flash",
    input="Explain artificial intelligence in simple words."
)

# Print the response
print(response.output_text)