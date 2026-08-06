"""
assistant.py — Drawing Parameter Optimizer
==========================================
Two modes:
  1. Default params (always available, instant): well-tuned per-style defaults.
  2. Gemini Vision (optional, when API key works): real image analysis using the model.

The fake "Local AI analyzer" (edge_density heuristics) has been removed.
It was generating plausible-looking numbers with no real impact on quality.
"""

import os
import io
import json
from PIL import Image
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# gemini-3.1-flash-lite is confirmed available via models.list()
GEMINI_MODEL = "gemini-3.1-flash-lite"

client = None
try:
    from google import genai
    if os.environ.get("GEMINI_API_KEY"):
        client = genai.Client()
except ImportError:
    pass



class DrawingParams(BaseModel):
    blur_size: int = Field(description="Gaussian blur radius, ODD integer 1-21.")
    threshold_block: int = Field(description="Adaptive threshold block size, ODD integer 3-51.")
    threshold_c: int = Field(description="Canny threshold control 1-20. Higher = fewer lines.")
    jitter: float = Field(description="Hand-drawn jitter 0.0-1.5. 0.1=precise, 0.5=natural.")
    hatching: float = Field(description="Shadow hatching 0.0-0.3. Use 0.0 for all except Realistic Sketch.")
    bg_color_wash: bool = Field(description="Color underpainting layer.")
    wash_opacity: int = Field(description="Underpainting opacity 0-150.")
    sketch_opacity: float = Field(description="Faint guideline opacity 0.0-0.25.")
    line_art_width: int = Field(description="Stroke width: 1=pencil, 2=anime, 3=manga.")
    explanation: str = Field(description="Why these values were chosen.")


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
        "explanation": "Anime: bilateral-smoothed Canny, clean sparse outlines, no wash."
    },
    "Realistic Sketch": {
        "blur_size": 3, "threshold_block": 11, "threshold_c": 5,
        "jitter": 0.40, "hatching": 0.0, "bg_color_wash": True,
        "wash_opacity": 50, "sketch_opacity": 0.12, "line_art_width": 1,
        "explanation": "Realistic Sketch: thin tonal charcoal strokes on warm paper."
    },
    "Colored Pencil Sketch": {
        "blur_size": 3, "threshold_block": 11, "threshold_c": 4,
        "jitter": 0.35, "hatching": 0.0, "bg_color_wash": True,
        "wash_opacity": 75, "sketch_opacity": 0.13, "line_art_width": 1,
        "explanation": "Colored Pencil: color-sampled strokes with soft wash underpainting."
    },
    "Oil Painting": {
        "blur_size": 7, "threshold_block": 25, "threshold_c": 8,
        "jitter": 0.65, "hatching": 0.0, "bg_color_wash": False,
        "wash_opacity": 0, "sketch_opacity": 0.0, "line_art_width": 1,
        "explanation": "Oil Painting: directional impasto brush strokes, canvas background."
    },
    "Paint-by-Numbers Blueprint": {
        "blur_size": 5, "threshold_block": 25, "threshold_c": 4,
        "jitter": 0.1, "hatching": 0.0, "bg_color_wash": False,
        "wash_opacity": 0, "sketch_opacity": 0.0, "line_art_width": 1,
        "explanation": "Paint-by-Numbers: flat color fills, numbered regions, clean borders."
    },
}


def get_default_parameters(vibe_style: str) -> dict:
    """Return well-tuned defaults for the given style. No API calls."""
    return dict(_DEFAULTS.get(vibe_style, _DEFAULTS["Realistic Sketch"]))


# ─────────────────────────────────────────────────────────────────────────────
#  Gemini Vision optimizer (only called when user explicitly enables it)
# ─────────────────────────────────────────────────────────────────────────────

def get_optimized_parameters(
    pil_img: Image.Image,
    vibe_style: str,
    model_name: str = None,
    force_gemini: bool = False
) -> dict:
    """
    Returns drawing parameters for the given image + style.

    force_gemini=False (default): returns hardcoded style defaults immediately.
      → No API calls, no latency, no rate limits. Drawing starts instantly.

    force_gemini=True: tries each Gemini key in turn. On 429/error, falls back
      to style defaults silently. Drawing never blocked.
    """
    # Always start with the style defaults
    defaults = get_default_parameters(vibe_style)

    if not force_gemini:
        return defaults

    # ── Gemini refinement (optional) ──
    keys = get_all_api_keys()
    if not keys:
        print("[Gemini] No API keys. Using style defaults.")
        return defaults

    # Always use gemini-2.0-flash-lite
    model = GEMINI_MODEL

    style_rules = {
        "Anime Outline":
            "Clean sparse outlines only. threshold_c=8-14, line_art_width=2, bg_color_wash=false, hatching=0.",
        "Realistic Sketch":
            "Canny-only structural edges. threshold_c=3-7, jitter=0.3-0.55, line_art_width=1, hatching=0-0.15.",
        "Colored Pencil Sketch":
            "Color-sampled pencil strokes. threshold_c=2-5, bg_color_wash=true, wash_opacity=60-110, hatching=0.",
        "Oil Painting":
            "Painterly impasto. blur_size=7-13, jitter=0.5-1.0, bg_color_wash=false, hatching=0.",
        "Paint-by-Numbers Blueprint":
            "Flat region outlines. threshold_c=3-6, line_art_width=1, bg_color_wash=false, hatching=0.",
    }.get(vibe_style, "")

    prompt = (
        f"You are a master artist. Analyze this image for the '{vibe_style}' drawing style.\n"
        f"Style rules: {style_rules}\n"
        f"Parameters to return (JSON):\n"
        f"  blur_size (ODD 1-21), threshold_block (ODD 3-51), threshold_c (1-20),\n"
        f"  jitter (0.0-1.5), hatching (0.0-0.3), bg_color_wash (bool),\n"
        f"  wash_opacity (0-150), sketch_opacity (0.0-0.25), line_art_width (1-3),\n"
        f"  explanation (short string).\n"
        f"Base your choices on the image's complexity, lighting, edge density, and color."
    )

    for key in keys:
        try:
            from google import genai
            from google.genai import types

            temp_client = genai.Client(api_key=key)

            # 256px thumbnail → minimal token cost
            thumb = pil_img.copy()
            thumb.thumbnail((256, 256), Image.Resampling.BILINEAR)

            response = temp_client.models.generate_content(
                model=model,
                contents=[thumb, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DrawingParams,
                    temperature=0.15,
                )
            )
            thumb.close()

            p = json.loads(response.text)

            # Enforce ODD for OpenCV params
            for f in ("blur_size", "threshold_block"):
                v = int(p.get(f, 5))
                p[f] = v if v % 2 == 1 else max(1, v - 1)

            # Safety clamps
            p["hatching"] = max(0.0, min(0.3, float(p.get("hatching", 0.0))))
            if vibe_style in ("Anime Outline", "Oil Painting", "Paint-by-Numbers Blueprint"):
                p["hatching"] = 0.0
                p["bg_color_wash"] = False

            tag = f"Gemini ({model}, key ...{key[-6:]})"
            p["explanation"] = f"{tag}: {str(p.get('explanation', ''))[:120]}"
            print(f"[Gemini] OK: {p['explanation'][:80]}")
            return p

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"[Gemini] Key ...{key[-6:]} rate-limited, trying next.")
            else:
                print(f"[Gemini] Key ...{key[-6:]} error: {type(e).__name__}: {err[:80]}")
            continue

    print("[Gemini] All keys failed. Using style defaults.")
    return defaults
