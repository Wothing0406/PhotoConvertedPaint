"""
processor.py — Drawing Mode Dispatcher
=======================================
This file is a thin router. Each vibe has its own complete
implementation in src/modes/:

  Realistic Sketch          → src/modes/sketch.py
  Colored Pencil Sketch     → src/modes/pencil.py
  Anime Outline             → src/modes/anime.py
  Oil Painting              → src/modes/oil.py
  Paint-by-Numbers Blueprint→ src/modes/blueprint.py

All mode generators follow the contract:
  draw(pil_img, **params) → Generator[PIL.Image.Image, None, None]
  Every yielded frame is a .copy() of the live canvas — never the
  canvas itself — so callers can safely .close() each frame.
"""

import os
from PIL import Image
from rembg import remove

from src.modes import sketch, pencil, anime, oil, blueprint
from src.gpu_utils import gpu_status, GPU_AVAILABLE, gpu_clear_cache


VIBE_MAP = {
    "Realistic Sketch":            sketch.draw,
    "Colored Pencil Sketch":       pencil.draw,
    "Anime Outline":               anime.draw,
    "Oil Painting":                oil.draw,
    "Paint-by-Numbers Blueprint":  blueprint.draw,
}


# Pre-initialize rembg session on GPU if CUDA is available
import rembg
try:
    # Use CUDA execution provider for RTX 3050
    gpu_session = rembg.new_session(
        model_name="u2net",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print("[Processor] Rembg GPU (CUDA) session initialized successfully!")
except Exception as e:
    print(f"[Processor] GPU initialization failed ({e}). Falling back to CPU.")
    gpu_session = rembg.new_session(model_name="u2net")

print(f"[Processor] Drawing engine: {gpu_status()}")


def remove_background_ai(pil_img: Image.Image) -> Image.Image:
    return rembg.remove(pil_img, session=gpu_session)



def progressive_draw_generator(
    pil_img: Image.Image,
    vibe: str = "Realistic Sketch",
    remove_bg: bool = True,
    blur_size: int = 3,
    threshold_block: int = 11,
    threshold_c: int = 5,
    jitter: float = 0.40,
    hatching_intensity: float = 0.0,
    batch_size: int = 10,
    session_id: str = None,
    bg_color_wash: bool = True,
    wash_opacity: int = 60,
    sketch_opacity: float = 0.12,
    line_art_width: int = 1,
    blending_radius: int = 25,
    shadow_strength: float = 0.35
):
    """
    Main entry point called by app.py /api/draw-stream.
    Resizes the image, optionally removes background, then delegates
    to the appropriate mode generator.

    All yielded frames are PIL.Image copies (RGB or RGBA).
    The generator always runs to completion — no early-stop mechanism.
    """
    import cv2
    import numpy as np

    # Clear VRAM at start to ensure maximum free capacity
    gpu_clear_cache()

    # ── Defensive normalise: always give modes clean RGB (H,W,3) uint8 ──────────
    # Handles CMYK, P (palette), L (grayscale), LA, RGBA, and exotic formats
    if pil_img.mode not in ("RGB",):
        pil_img = pil_img.convert("RGBA").convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")

    # ── High Resolution Drawing: Cap at 1200px to prevent Out of Memory ──────
    # 1200px is highly detailed (1.2 Megapixels) but uses ~45% less VRAM than 1600px
    max_res = 1200
    w0, h0 = pil_img.size
    scale = min(1.0, max_res / max(w0, h0))
    if scale < 1.0:
        pil_img = pil_img.resize(
            (int(w0 * scale), int(h0 * scale)), Image.Resampling.LANCZOS
        )
    print(f"[Processor] Resolution optimized to {pil_img.size} for high-fidelity drawing without memory overflow")



    # ── Background removal (optional) ─────────────────────────────────────────
    if remove_bg:
        try:
            removed = remove_background_ai(pil_img)
            arr = np.array(removed)
            # Only keep removal result if subject is meaningful (>2% opacity)
            if arr.ndim == 3 and arr.shape[2] == 4:
                w, h = pil_img.size
                if np.sum(arr[:, :, 3] > 10) > w * h * 0.02:
                    # Use a warm, neutral paper color background (240, 238, 233) instead of harsh pure white.
                    # This reduces border contrast spikes so edge detectors don't lose internal details.
                    bg = Image.new("RGB", removed.size, (240, 238, 233))
                    
                    # Smooth the alpha mask edge slightly using PIL blur to prevent harsh cuts
                    alpha = removed.split()[3]
                    smoothed_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.0))
                    
                    bg.paste(removed, mask=smoothed_alpha)
                    pil_img = bg
        except Exception as e:
            print(f"[rembg] Background removal failed: {e}")

    # ── Dispatch to mode ──────────────────────────────────────────────────────
    draw_fn = VIBE_MAP.get(vibe)
    if draw_fn is None:
        # Unknown vibe: fall back to Realistic Sketch
        print(f"[processor] Unknown vibe '{vibe}', falling back to Realistic Sketch.")
        draw_fn = sketch.draw

    params = dict(
        blur_size=blur_size,
        threshold_block=threshold_block,
        threshold_c=threshold_c,
        jitter=jitter,
        hatching=hatching_intensity,
        batch_size=batch_size,
        bg_color_wash=bg_color_wash,
        wash_opacity=wash_opacity,
        sketch_opacity=sketch_opacity,
        line_art_width=line_art_width,
        shadow_strength=shadow_strength,
    )

    # Video recording (optional)
    video_writer = None
    if session_id:
        try:
            w, h = pil_img.size
            vw = w - (w % 2)
            vh = h - (h % 2)
            video_path = os.path.join("output", f"drawing_{session_id}.mp4")
            
            # Write raw video using mp4v (fast, reliable in OpenCV)
            # We will post-process this to H.264 using FFmpeg before final export
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                15.0, (vw, vh)
            )
            # Test write ability
            if not video_writer.isOpened():
                print("[video] Failed to initialize video writer.")
                video_writer = None
        except Exception as e:
            print(f"[video] Could not init writer: {e}")
            video_writer = None

    try:
        for frame in draw_fn(pil_img, **params):
            # Write to video
            if video_writer is not None:
                try:
                    arr = np.array(frame.convert("RGB"))
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    # Use local vw and vh defined earlier (OpenCV VideoWriter.get does not support CAP_PROP_FRAME_WIDTH)
                    if bgr.shape[1] != vw or bgr.shape[0] != vh:
                        bgr = cv2.resize(bgr, (vw, vh))
                    video_writer.write(bgr)
                except Exception as ve:
                    print(f"[video] Failed to write frame: {ve}")
            yield frame
    finally:
        if video_writer is not None:
            video_writer.release()
            print("[video] Writer released.")
        # Clear VRAM cache immediately after drawing finishes
        gpu_clear_cache()
