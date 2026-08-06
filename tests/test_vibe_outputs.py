import os
import sys
from PIL import Image, ImageDraw

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.processor import progressive_draw_generator
from src.exporter import save_png_optimized

def create_detailed_test_image(path: str):
    """
    Creates a detailed synthetic image simulating a portrait in a frame
    with eyes, nose, mouth, bowtie, and a textured background.
    """
    # 400x500 test image
    img = Image.new("RGB", (400, 500), (240, 230, 220)) # Light background
    draw = ImageDraw.Draw(img)
    
    # Draw dark curtains/background frame
    draw.rectangle([20, 20, 380, 480], outline=(80, 50, 30), width=15)
    
    # Draw body / coat (teal)
    draw.polygon([(100, 500), (200, 350), (300, 500)], fill=(20, 100, 120))
    
    # Draw bowtie (gold)
    draw.polygon([(170, 350), (200, 370), (170, 390)], fill=(210, 160, 40))
    draw.polygon([(230, 350), (200, 370), (230, 390)], fill=(210, 160, 40))
    draw.ellipse([190, 360, 210, 380], fill=(210, 160, 40))
    
    # Draw head (husky gray/white)
    draw.ellipse([120, 180, 280, 340], fill=(120, 120, 125)) # Head shape
    draw.polygon([(120, 180), (150, 100), (180, 200)], fill=(80, 80, 80)) # Left Ear
    draw.polygon([(280, 180), (250, 100), (220, 200)], fill=(80, 80, 80)) # Right Ear
    draw.ellipse([150, 240, 250, 340], fill=(245, 245, 245)) # Muzzle shape (white)
    
    # Face features (eyes, nose, mouth) - crucial details that should not be missing
    draw.ellipse([160, 220, 180, 240], fill=(20, 120, 240)) # Left Eye (Blue)
    draw.ellipse([220, 220, 240, 240], fill=(160, 80, 20)) # Right Eye (Brown)
    draw.polygon([(185, 260), (215, 260), (200, 285)], fill=(15, 10, 10)) # Black Nose
    
    # Mouth lines
    draw.arc([170, 280, 200, 310], start=0, end=180, fill=(15, 10, 10), width=3)
    draw.arc([200, 280, 230, 310], start=0, end=180, fill=(15, 10, 10), width=3)
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    print(f"Synthetic test image created at {path}")

def run_vibe_tests():
    test_img_path = "test_images/husky_mock.png"
    create_detailed_test_image(test_img_path)
    
    input_img = Image.open(test_img_path)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    vibes = [
        "Realistic Sketch", 
        "Anime Outline", 
        "Colored Pencil Sketch", 
        "Paint-by-Numbers Blueprint"
    ]
    
    for vibe in vibes:
        print(f"\n--- Testing Vibe: {vibe} ---")
        # Run drawing generator
        frames = list(progressive_draw_generator(
            input_img,
            vibe=vibe,
            remove_bg=False, # Keep background frame and curtains
            blur_size=3,
            threshold_block=11,
            threshold_c=2,
            jitter=0.4,
            hatching_intensity=0.6,
            batch_size=50
        ))
        
        # Save final frame
        final_frame = frames[-1]
        filename = f"test_{vibe.lower().replace(' ', '_').replace('-', '_')}.png"
        output_path = os.path.join(output_dir, filename)
        save_png_optimized(final_frame, output_path)
        print(f"Result saved to {output_path} (Frames generated: {len(frames)})")

if __name__ == "__main__":
    run_vibe_tests()
