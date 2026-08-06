import os
import sys
import unittest
from PIL import Image

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import load_image
from src.processor import extract_contours, progressive_draw_generator
from src.exporter import save_png_optimized

class TestTranhVeCore(unittest.TestCase):
    def setUp(self):
        # Create a simple 200x200 pixel test image (a black circle on white background)
        self.test_img_path = "test_images/dummy_test.png"
        if not os.path.exists("test_images"):
            os.makedirs("test_images")
            
        img = Image.new("RGBA", (200, 200), (255, 255, 255, 255))
        # Draw a simple shape to test contours
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([50, 50, 150, 150], fill=(0, 0, 0, 255))
        img.save(self.test_img_path)

    def test_image_loading(self):
        img = load_image(self.test_img_path)
        self.assertEqual(img.size, (200, 200))
        self.assertEqual(img.mode, "RGBA")

    def test_contour_extraction(self):
        img = load_image(self.test_img_path)
        contours, gray = extract_contours(img)
        self.assertTrue(len(contours) > 0)

    def test_progressive_drawing(self):
        img = load_image(self.test_img_path)
        frames = list(progressive_draw_generator(
            img, vibe="Monochrome", remove_bg=False, batch_size=5
        ))
        # Verify that we yielded multiple frames of progressive drawing
        self.assertTrue(len(frames) > 0)
        self.assertEqual(frames[-1].size, (200, 200))

    def test_exporter(self):
        img = load_image(self.test_img_path)
        output_path = "output/test_export.png"
        save_png_optimized(img, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Verify saved image is readable
        saved_img = Image.open(output_path)
        self.assertEqual(saved_img.size, (200, 200))

    def tearDown(self):
        # Clean up test output
        if os.path.exists(self.test_img_path):
            os.remove(self.test_img_path)
        if os.path.exists("output/test_export.png"):
            os.remove("output/test_export.png")

if __name__ == "__main__":
    unittest.main()
