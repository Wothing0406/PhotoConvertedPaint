import os
import io
import base64
import uvicorn
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from src.processor import progressive_draw_generator
from src.exporter import save_png_optimized
from src.assistant import get_optimized_parameters, is_api_available

import time
import zipfile
import numpy as np

app = FastAPI(title="TranhVe API")

# Ensure output directory exists
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def get_index():
    """
    Serves the custom frontend interface.
    """
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>static/index.html not found</h1>", status_code=404)

@app.get("/api/api-status")
def api_status():
    """
    Returns True if Gemini API is available.
    """
    return {"available": is_api_available()}

@app.post("/api/optimize-params")
async def optimize_params(
    image: UploadFile,
    vibe: str = Form(...),
    force_gemini: bool = Form(False)
):
    """
    Always runs Local Vision Analyzer first (instant, 0 tokens).
    Only tries Gemini if force_gemini=True AND API keys are available.
    Drawing is never blocked by Gemini rate limits.
    """
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        params = get_optimized_parameters(pil_img, vibe_style=vibe, force_gemini=force_gemini)
        return params
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/quality-check")
async def quality_check(
    original: UploadFile,
    result_path: str = Form(...)
):
    """
    Quality Check step: Compares the drawn result PNG to the original image.
    Returns a quality score (0-100) based on edge coverage and structural similarity.
    """
    try:
        import cv2
        
        # Read original
        orig_contents = await original.read()
        orig_img = Image.open(io.BytesIO(orig_contents)).convert("RGB")
        orig_np = np.array(orig_img)
        
        # Read result - extract only the filename to secure against path traversal
        filename = os.path.basename(result_path)
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(full_path):
            # Fallback: scan output dir for matching filename
            found = None
            for f in os.listdir(OUTPUT_DIR):
                if f == filename or (filename.startswith("final_") and f == filename):
                    found = os.path.join(OUTPUT_DIR, f)
                    break
            if found and os.path.exists(found):
                full_path = found
            else:
                raise HTTPException(status_code=404, detail=f"Result file not found: {filename}. Please draw first and wait for completion.")
        
        result_img = Image.open(full_path).convert("RGB")
        result_np = np.array(result_img)
        
        # Resize to same size for comparison
        target_h, target_w = min(orig_np.shape[0], 400), min(orig_np.shape[1], 400)
        orig_resized = cv2.resize(orig_np, (target_w, target_h))
        result_resized = cv2.resize(result_np, (target_w, target_h))
        
        # 1. Edge coverage: how many original edges are present in result
        orig_gray = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2GRAY)
        result_gray = cv2.cvtColor(result_resized, cv2.COLOR_RGB2GRAY)
        
        orig_edges = cv2.Canny(cv2.GaussianBlur(orig_gray, (3,3), 0), 50, 150)
        result_edges = cv2.Canny(cv2.GaussianBlur(result_gray, (3,3), 0), 50, 150)
        
        # Dilate edges slightly to allow for positional jitter
        kernel = np.ones((3,3), np.uint8)
        orig_edges_dilated = cv2.dilate(orig_edges, kernel, iterations=2)
        
        orig_edge_pixels = np.sum(orig_edges > 0)
        if orig_edge_pixels > 0:
            matched = np.sum((result_edges > 0) & (orig_edges_dilated > 0))
            edge_coverage = min(100, int(matched * 100 / orig_edge_pixels))
        else:
            edge_coverage = 50
        
        # 2. Structural completeness: check if result has non-white pixels in key regions
        result_non_white = np.sum(result_gray < 240)
        result_total = target_w * target_h
        coverage_ratio = min(100, int(result_non_white * 100 / result_total))
        
        # 3. Detail density: number of unique edges in result relative to original
        result_edge_count = np.sum(result_edges > 0)
        orig_edge_count = max(1, orig_edge_pixels)
        detail_ratio = min(100, int(result_edge_count * 100 / orig_edge_count))
        
        # Composite quality score
        quality_score = int(
            edge_coverage * 0.4 +     # How well edges match
            coverage_ratio * 0.3 +    # How much of canvas is covered
            detail_ratio * 0.3        # Detail density match
        )
        
        grade = "Xuất sắc" if quality_score >= 80 else \
                "Tốt" if quality_score >= 60 else \
                "Khá" if quality_score >= 40 else "Cần cải thiện"
        
        suggestions = []
        if edge_coverage < 50:
            suggestions.append("Giảm threshold_c để bắt nhiều nét hơn")
        if detail_ratio < 40:
            suggestions.append("Giảm blur_size để giữ chi tiết mịn")
        if coverage_ratio < 30:
            suggestions.append("Tăng batch_size hoặc chờ vẽ xong hoàn toàn")
        
        return {
            "quality_score": quality_score,
            "grade": grade,
            "edge_coverage_pct": edge_coverage,
            "canvas_coverage_pct": coverage_ratio,
            "detail_density_pct": detail_ratio,
            "suggestions": suggestions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")

def cleanup_old_files(max_age_seconds: int = 300):
    """
    Deletes files in OUTPUT_DIR older than max_age_seconds (5 minutes) to prevent disk buildup.
    """
    try:
        now = time.time()
        for filename in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    print(f"Cleaned up cached file: {file_path}")
    except Exception as e:
        print(f"Error during cache cleanup: {e}")

@app.post("/api/draw-stream")
async def draw_stream(
    image: UploadFile,
    vibe: str = Form(...),
    remove_bg: bool = Form(True),
    blur: int = Form(5),
    thresh_block: int = Form(11),
    thresh_c: int = Form(2),
    jitter: float = Form(0.5),
    hatching: float = Form(0.5),
    speed: int = Form(20),
    bg_color_wash: bool = Form(True),
    wash_opacity: int = Form(75),
    sketch_opacity: float = Form(0.15),
    line_art_width: int = Form(1)
):
    """
    Processes the image and yields progressive JPEG-encoded frame chunks.
    Saves drawing video and packages the final result into a ZIP file.
    Streams progress as: data: <base64_jpeg>\\n\\n
    Streams totals as:   data: total:<N>\\n\\n
    Streams filepath as: data: filepath:<path>\\n\\n
    """
    cleanup_old_files()

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        def error_generator():
            yield f"data: error:Failed to parse image: {str(e)}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    # Store file bytes for quality check later
    image_bytes = contents

    def event_generator():
        session_id = str(int(time.time() * 1000))
        try:
            final_frame = None
            frame_count = 0

            for frame in progressive_draw_generator(
                pil_img,
                vibe=vibe,
                remove_bg=remove_bg,
                blur_size=blur,
                threshold_block=thresh_block,
                threshold_c=thresh_c,
                jitter=jitter,
                hatching_intensity=hatching,
                batch_size=speed,
                session_id=session_id,
                bg_color_wash=bg_color_wash,
                wash_opacity=wash_opacity,
                sketch_opacity=sketch_opacity,
                line_art_width=line_art_width
            ):
                if final_frame is not None:
                    final_frame.close()
                final_frame = frame.copy()
                frame_count += 1

                # Encode as JPEG (10x smaller than PNG) for faster streaming
                buffered = io.BytesIO()
                frame.convert("RGB").save(buffered, format="JPEG", quality=75)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                yield f"data: {img_str}\n\n"
                frame.close()

            # Once finished, save final optimized PNG and create ZIP
            if final_frame is not None:
                image_filepath = os.path.join(OUTPUT_DIR, f"final_{session_id}.png")
                save_png_optimized(final_frame, image_filepath)
                
                video_filepath = os.path.join(OUTPUT_DIR, f"drawing_{session_id}.mp4")
                
                # Convert raw OpenCV video to highly compatible H.264 using FFmpeg (for iOS/Android/Web)
                if os.path.exists(video_filepath):
                    import subprocess
                    h264_filepath = os.path.join(OUTPUT_DIR, f"drawing_{session_id}_h264.mp4")
                    try:
                        # -y to overwrite, -vcodec libx264 for universal mobile compatibility, -pix_fmt yuv420p for Safari/iOS
                        result = subprocess.run(
                            ["ffmpeg", "-y", "-i", video_filepath, "-vcodec", "libx264", "-pix_fmt", "yuv420p", h264_filepath],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0 and os.path.exists(h264_filepath):
                            os.remove(video_filepath)
                            os.rename(h264_filepath, video_filepath)
                    except Exception as fe:
                        print(f"[FFmpeg] Exception encoding video: {fe}")
                
                zip_filepath = os.path.join(OUTPUT_DIR, f"tranhve_{session_id}.zip")
                with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                    if os.path.exists(image_filepath):
                        zipf.write(image_filepath, arcname="final_artwork.png")
                    if os.path.exists(video_filepath):
                        zipf.write(video_filepath, arcname="drawing_process.mp4")
                
                yield f"data: done:{frame_count}\n\n"
                yield f"data: filepath:{zip_filepath}\n\n"

        except Exception as e:
            yield f"data: error:Error processing drawing: {str(e)}\n\n"


    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/download")
def download_image(path: str):
    """
    Secure download endpoint checking path traversal.
    """
    normalized_path = os.path.normpath(path)
    if not normalized_path.startswith("output"):
        raise HTTPException(status_code=400, detail="Access denied. Path traversal blocked.")
    
    if os.path.exists(normalized_path):
        return FileResponse(normalized_path, media_type="application/zip", filename="tranhve_artwork.zip")
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
