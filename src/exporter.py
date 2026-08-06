import os
from PIL import Image

def save_png_optimized(pil_img: Image.Image, output_path: str, compress_level: int = 9) -> str:
    """
    Saves a PIL Image to output_path as an optimized PNG.
    Strips EXIF data for privacy and compresses losslessly.
    """
    # Ensure directory exists
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    # Strip metadata by copying the image and clearing its info dictionary
    clean_img = pil_img.copy()
    clean_img.info = {}
    
    # Save optimized
    clean_img.save(
        output_path, 
        format="PNG", 
        optimize=True, 
        compress_level=compress_level
    )
    return output_path
