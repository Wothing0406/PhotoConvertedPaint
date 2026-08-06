import os
import requests
import zipfile

def test_api_status():
    print("Testing /api/api-status...")
    r = requests.get("http://localhost:7860/api/api-status")
    print(f"Status: {r.status_code}, Response: {r.json()}")
    assert r.status_code == 200

def test_draw_stream():
    print("\nTesting /api/draw-stream...")
    # Create a small dummy image
    from PIL import Image
    import io
    img = Image.new("RGB", (200, 200), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    
    files = {"image": ("dummy.png", img_byte_arr, "image/png")}
    data = {
        "vibe": "Realistic Sketch",
        "remove_bg": "false",
        "blur": "3",
        "thresh_block": "11",
        "thresh_c": "2",
        "jitter": "0.1",
        "hatching": "0.0",
        "speed": "50"
    }
    
    r = requests.post("http://localhost:7860/api/draw-stream", files=files, data=data, stream=True)
    assert r.status_code == 200
    
    zip_path = None
    frame_count = 0
    
    for line in r.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data:"):
                payload = decoded[5:].strip()
                if payload.startswith("filepath:"):
                    zip_path = payload[9:].strip()
                elif payload.startswith("error:"):
                    print(f"Server Error: {payload}")
                else:
                    frame_count += 1
                    
    print(f"Frames received: {frame_count}")
    print(f"ZIP path returned: {zip_path}")
    
    assert zip_path is not None
    assert zip_path.endswith(".zip")
    assert os.path.exists(zip_path)
    
    # Verify ZIP contents
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        namelist = zipf.namelist()
        print(f"ZIP namelist: {namelist}")
        assert "final_artwork.png" in namelist
        assert "drawing_process.mp4" in namelist
        
    print("FastAPI Stream & ZIP packaging tests passed successfully!")

if __name__ == "__main__":
    try:
        test_api_status()
        test_draw_stream()
    except Exception as e:
        print(f"Tests failed: {e}")
