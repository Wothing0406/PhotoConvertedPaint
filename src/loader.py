import os
import requests
from io import BytesIO
from PIL import Image

def load_image(source: str) -> Image.Image:
    """
    Loads an image from a URL or a local file path.
    Converts it to RGBA format.
    """
    if not source:
        raise ValueError("Source path or URL cannot be empty.")
    
    # Check if the source is a URL
    if source.startswith("http://") or source.startswith("https://"):
        try:
            response = requests.get(source, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            raise IOError(f"Failed to download image from URL '{source}': {e}")
    else:
        # Load from local path
        if not os.path.exists(source):
            raise FileNotFoundError(f"Local file not found at path '{source}'")
        try:
            image = Image.open(source)
        except Exception as e:
            raise IOError(f"Failed to open local image file '{source}': {e}")
            
    # Convert image to RGBA (Red, Green, Blue, Alpha) to ensure alpha channel compatibility
    return image.convert("RGBA")
