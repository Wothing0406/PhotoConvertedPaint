"""
assistant.py — Drawing Parameter Optimizer
==========================================
Two modes:
  1. Default params (always available, instant): well-tuned per-style defaults.
  2. Gemini Vision (optional, when API key works): real image analysis using the model.
     Integrates semantic facial landmark detection (landmarks, axis, head tilt)
     to guide the structural draft layer and define the visual focal point.
"""

import os
import io
import json
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_MODEL = "gemini-3.1-flash-lite"

client = None
try:
    from google import genai
    if os.environ.get("GEMINI_API_KEY"):
        client = genai.Client()
except ImportError:
    pass


class DrawingParams(BaseModel):
    image_subject: str = Field(description="Detected category of the image. Must be one of: 'portrait_human', 'animal_pet', 'landscape_nature', 'object_still_life'.")
    
    # ── Semantic Landmark Fields to guide drawing loops ─────────────────────
    face_center_x: float = Field(description="Fractional center of the main subject/face horizontally (0.0 to 1.0). Default 0.5.")
    face_center_y: float = Field(description="Fractional center of the main subject/face vertically (0.0 to 1.0). Default 0.4.")
    face_width: float = Field(description="Fractional width of the face (0.0 to 1.0). Default 0.3.")
    face_height: float = Field(description="Fractional height of the face (0.0 to 1.0). Default 0.45.")
    eye_left_x: float = Field(description="Fractional left eye X coordinate (0.0 to 1.0).")
    eye_left_y: float = Field(description="Fractional left eye Y coordinate (0.0 to 1.0).")
    eye_right_x: float = Field(description="Fractional right eye X coordinate (0.0 to 1.0).")
    eye_right_y: float = Field(description="Fractional right eye Y coordinate (0.0 to 1.0).")
    head_tilt_angle: float = Field(description="Angle of head tilt in degrees (-45.0 to 45.0). Positive means tilted right, negative left.")
    
    explanation: str = Field(description="Description of the subject detected and rationale for landmarks.")


def is_api_available() -> bool:
    key = os.environ.get("GEMINI_API_KEY", "")
    return client is not None and len(key.strip()) > 0


def get_all_api_keys() -> list:
    keys = []
    main = os.environ.get("GEMINI_API_KEY", "").strip()
    if main:
        keys.append(main)
    idx = 2
    while True:
        k = os.environ.get(f"GEMINI_API_KEY_{idx}", "").strip()
        if k:
            keys.append(k)
            idx += 1
        else:
            break
    return keys


# ─────────────────────────────────────────────────────────────────────────────
#  Well-tuned defaults (replaces the fake "Local AI" heuristics)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "Anime Outline": {
        "blur_size": 5, "threshold_block": 21, "threshold_c": 10,
        "jitter": 0.15, "hatching": 0.0, "bg_color_wash": False,
        "wash_opacity": 0, "sketch_opacity": 0.0, "line_art_width": 2,
        "shadow_strength": 0.0, "image_subject": "portrait_human",
        "face_center_x": 0.5, "face_center_y": 0.4, "face_width": 0.3, "face_height": 0.45,
        "eye_left_x": 0.43, "eye_left_y": 0.38, "eye_right_x": 0.57, "eye_right_y": 0.38,
        "head_tilt_angle": 0.0,
        "explanation": "Anime: bilateral-smoothed Canny, clean sparse outlines, no wash."
    },
    "Realistic Sketch": {
        "blur_size": 3, "threshold_block": 11, "threshold_c": 5,
        "jitter": 0.40, "hatching": 0.0, "bg_color_wash": True,
        "wash_opacity": 50, "sketch_opacity": 0.12, "line_art_width": 1,
        "shadow_strength": 0.35, "image_subject": "portrait_human",
        "face_center_x": 0.5, "face_center_y": 0.4, "face_width": 0.3, "face_height": 0.45,
        "eye_left_x": 0.43, "eye_left_y": 0.38, "eye_right_x": 0.57, "eye_right_y": 0.38,
        "head_tilt_angle": 0.0,
        "explanation": "Realistic Sketch: thin tonal charcoal strokes on warm paper."
    },
    "Colored Pencil Sketch": {
        "blur_size": 3, "threshold_block": 11, "threshold_c": 4,
        "jitter": 0.35, "hatching": 0.0, "bg_color_wash": True,
        "wash_opacity": 75, "sketch_opacity": 0.13, "line_art_width": 1,
        "shadow_strength": 0.0, "image_subject": "portrait_human",
        "face_center_x": 0.5, "face_center_y": 0.4, "face_width": 0.3, "face_height": 0.45,
        "eye_left_x": 0.43, "eye_left_y": 0.38, "eye_right_x": 0.57, "eye_right_y": 0.38,
        "head_tilt_angle": 0.0,
        "explanation": "Colored Pencil: color-sampled strokes with soft wash underpainting."
    },
    "Oil Painting": {
        "blur_size": 7, "threshold_block": 25, "threshold_c": 8,
        "jitter": 0.65, "hatching": 0.0, "bg_color_wash": False,
        "wash_opacity": 0, "sketch_opacity": 0.0, "line_art_width": 1,
        "shadow_strength": 0.0, "image_subject": "portrait_human",
        "face_center_x": 0.5, "face_center_y": 0.4, "face_width": 0.3, "face_height": 0.45,
        "eye_left_x": 0.43, "eye_left_y": 0.38, "eye_right_x": 0.57, "eye_right_y": 0.38,
        "head_tilt_angle": 0.0,
        "explanation": "Oil Painting: directional impasto brush strokes, canvas background."
    },
    "Paint-by-Numbers Blueprint": {
        "blur_size": 5, "threshold_block": 25, "threshold_c": 4,
        "jitter": 0.1, "hatching": 0.0, "bg_color_wash": False,
        "wash_opacity": 0, "sketch_opacity": 0.0, "line_art_width": 1,
        "shadow_strength": 0.0, "image_subject": "portrait_human",
        "face_center_x": 0.5, "face_center_y": 0.4, "face_width": 0.3, "face_height": 0.45,
        "eye_left_x": 0.43, "eye_left_y": 0.38, "eye_right_x": 0.57, "eye_right_y": 0.38,
        "head_tilt_angle": 0.0,
        "explanation": "Paint-by-Numbers: flat color fills, numbered regions, clean borders."
    },
}


def get_default_parameters(vibe_style: str) -> dict:
    return dict(_DEFAULTS.get(vibe_style, _DEFAULTS["Realistic Sketch"]))


def get_optimized_parameters(
    pil_img: Image.Image,
    vibe_style: str,
    model_name: str = None,
    force_gemini: bool = False
) -> dict:
    defaults = get_default_parameters(vibe_style)

    if not force_gemini:
        return defaults

    keys = get_all_api_keys()
    if not keys:
        print("[Gemini] No API keys. Using style defaults.")
        return defaults

    model = GEMINI_MODEL

    prompt = (
        f"You are a master artist analyzing an image to extract high-level semantic context and facial landmarks for a '{vibe_style}' drawing process.\n"
        f"\n"
        f"DIRECTIONS:\n"
        f"1. Detect the main subject type: 'portrait_human', 'animal_pet', 'landscape_nature', or 'object_still_life'.\n"
        f"2. If it is a portrait (human or animal), identify key facial coordinates as fractions of the image dimensions (0.0 to 1.0):\n"
        f"   - face_center_x, face_center_y: coordinate of the center of the face.\n"
        f"   - face_width, face_height: width and height of the face bounding box.\n"
        f"   - eye_left_x, eye_left_y, eye_right_x, eye_right_y: the coordinates of the eyes.\n"
        f"   - head_tilt_angle: angle of head tilt in degrees (-45.0 to 45.0). Positive = tilted right, negative = tilted left.\n"
        f"Return the exact fields defined in the schema."
    )

    for key in keys:
        try:
            from google import genai
            from google.genai import types

            temp_client = genai.Client(api_key=key)

            thumb = pil_img.copy()
            thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)

            import concurrent.futures
            
            def _api_call():
                return temp_client.models.generate_content(
                    model=model,
                    contents=[thumb, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DrawingParams,
                        temperature=0.15,
                    )
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_api_call)
                try:
                    response = future.result(timeout=8.0)
                except concurrent.futures.TimeoutError:
                    print(f"[Gemini] Key ...{key[-6:]} timed out.")
                    thumb.close()
                    continue

            thumb.close()
            p = json.loads(response.text)

            # Copy all landmark fields directly into the defaults dictionary
            for field in [
                "face_center_x", "face_center_y", "face_width", "face_height",
                "eye_left_x", "eye_left_y", "eye_right_x", "eye_right_y",
                "head_tilt_angle", "image_subject"
            ]:
                if field in p:
                    defaults[field] = p[field]

            subj = p.get("image_subject", "portrait_human")
            tag = f"Gemini ({model}, {subj})"
            defaults["explanation"] = f"{tag}: {str(p.get('explanation', ''))[:120]}"
            print(f"[Gemini] OK: {defaults['explanation'][:80]}")
            return defaults

        except Exception as e:
            err = str(e)
            print(f"[Gemini] Key error: {type(e).__name__}: {err[:80]}")
            continue

    print("[Gemini] All keys failed. Using style defaults.")
    return defaults
