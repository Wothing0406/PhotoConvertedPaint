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
    shadow_strength: float = Field(description="Intensity of shadow composite (0.0 to 0.5). For landscapes, set close to 0.0. For portraits, set 0.25-0.45. For animals/pets, set 0.15-0.30 to enhance depth without muddying fur textures.")
    image_subject: str = Field(description="Detected category of the image. Must be one of: 'portrait_human', 'animal_pet', 'landscape_nature', 'object_still_life'.")
    explanation: str = Field(description="Why these values were chosen. Start with '[Subject Type detected]' followed by rationale.")


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
        "explanation": "Anime: bilateral-smoothed Canny, clean sparse outlines, no wash."
    },
    "Realistic Sketch": {
        "blur_size": 3, "threshold_block": 11, "threshold_c": 5,
        "jitter": 0.40, "hatching": 0.0, "bg_color_wash": True,
        "wash_opacity": 50, "sketch_opacity": 0.12, "line_art_width": 1,
        "shadow_strength": 0.35, "image_subject": "portrait_human",
        "explanation": "Realistic Sketch: thin tonal charcoal strokes on warm paper."
    },
    "Colored Pencil Sketch": {
        "blur_size": 3, "threshold_block": 11, "threshold_c": 4,
        "jitter": 0.35, "hatching": 0.0, "bg_color_wash": True,
        "wash_opacity": 75, "sketch_opacity": 0.13, "line_art_width": 1,
        "shadow_strength": 0.0, "image_subject": "portrait_human",
        "explanation": "Colored Pencil: color-sampled strokes with soft wash underpainting."
    },
    "Oil Painting": {
        "blur_size": 7, "threshold_block": 25, "threshold_c": 8,
        "jitter": 0.65, "hatching": 0.0, "bg_color_wash": False,
        "wash_opacity": 0, "sketch_opacity": 0.0, "line_art_width": 1,
        "shadow_strength": 0.0, "image_subject": "portrait_human",
        "explanation": "Oil Painting: directional impasto brush strokes, canvas background."
    },
    "Paint-by-Numbers Blueprint": {
        "blur_size": 5, "threshold_block": 25, "threshold_c": 4,
        "jitter": 0.1, "hatching": 0.0, "bg_color_wash": False,
        "wash_opacity": 0, "sketch_opacity": 0.0, "line_art_width": 1,
        "shadow_strength": 0.0, "image_subject": "portrait_human",
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
            "Anime: Clean, smooth, and simplified cartoon line-art. Ensure threshold_c=8-15 to keep outlines clean and clear of background noise. Hatching=0, shadow_strength=0.0, bg_color_wash=false, line_art_width=2.",
        "Realistic Sketch":
            "Realistic Sketch: Soft tonal drawing with fine details. Use blur_size=3-5 for detailed textures like human/animal eyes, fur, or wrinkles. If the background is complex (textured/Van Gogh/illustrated), use threshold_c=6-11 to filter background clutter. Otherwise, use threshold_c=3-6. Shadow_strength should be 0.18-0.35 for depth.",
        "Colored Pencil Sketch":
            "Colored Pencil: Color-sampled lines with pencil texture and a warm underpainting wash. Use threshold_c=2-5. Keep wash_opacity=65-110. Hatching=0, shadow_strength=0.0, line_art_width=1.",
        "Oil Painting":
            "Oil Painting: Bold painterly brushstrokes. Use blur_size=7-13, jitter=0.5-0.9 to give randomized impasto brush weight. Hatching=0, shadow_strength=0.0, bg_color_wash=false.",
        "Paint-by-Numbers Blueprint":
            "Paint-by-Numbers: Outlined color-regions. Set threshold_c=3-6, line_art_width=1, bg_color_wash=false, hatching=0, shadow_strength=0.0.",
    }.get(vibe_style, "")

    prompt = (
        f"You are a master artist analyzing an image to prepare optimal drawing parameters for a '{vibe_style}' drawing process.\n"
        f"\n"
        f"PARAMETER DEFINITION & TUNING GUIDELINES:\n"
        f"- blur_size (ODD, 1-21): Lower values (3-5) preserve extremely fine details like eyelashes, catchlights in eyes, animal whiskers. High values (7-13) smooth out and simplify shapes.\n"
        f"- threshold_c (1-20): High value reduces details and filters noise. If background has complex patterns/strokes (e.g. Van Gogh style, busy textures), select a higher threshold_c (8-14) to avoid clutter. If the subject contains crucial fine lines, set threshold_c (2-6).\n"
        f"- wash_opacity (0-150): Control opacity of underpainting wash. 0 is pure white canvas.\n"
        f"- shadow_strength (0.0-0.5): Deepens dark tones. Keep very low (0.0-0.1) for landscapes to prevent blocky gray/black blobs, and moderate (0.2-0.4) for portraits to add facial depth.\n"
        f"\n"
        f"DIRECTIONS:\n"
        f"1. Detect the main subject: 'portrait_human', 'animal_pet', 'landscape_nature', or 'object_still_life'.\n"
        f"2. Check background complexity: detect if the background is complex (patterns, swirls, wallpaper, busy texture) or simple.\n"
        f"3. Apply specific style instructions: {style_rules}\n"
        f"Return the exact fields defined in the schema to produce the most artistic and detailed sketch output."
    )

    for key in keys:
        try:
            from google import genai
            from google.genai import types

            temp_client = genai.Client(api_key=key)

            # 256px thumbnail → minimal token cost
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
                    response = future.result(timeout=8.0)  # 8 seconds strict timeout
                except concurrent.futures.TimeoutError:
                    print(f"[Gemini] Key ...{key[-6:]} timed out, trying next.")
                    thumb.close()
                    continue

            thumb.close()
            p = json.loads(response.text)

            # Enforce ODD for OpenCV params
            for f in ("blur_size", "threshold_block"):
                v = int(p.get(f, 5))
                p[f] = v if v % 2 == 1 else max(1, v - 1)

            # Safety clamps
            p["hatching"] = max(0.0, min(0.3, float(p.get("hatching", 0.0))))
            p["shadow_strength"] = max(0.0, min(0.5, float(p.get("shadow_strength", 0.35))))
            
            # Subject-based constraints
            subj = p.get("image_subject", "portrait_human")
            if subj == "landscape_nature":
                p["shadow_strength"] = min(0.12, p["shadow_strength"])
            elif subj == "animal_pet":
                p["shadow_strength"] = min(0.28, max(0.12, p["shadow_strength"]))
                p["blur_size"] = min(5, p["blur_size"])  # Keep blur low for fur detail

            if vibe_style in ("Anime Outline", "Oil Painting", "Paint-by-Numbers Blueprint"):
                p["hatching"] = 0.0
                p["bg_color_wash"] = False
                p["shadow_strength"] = 0.0

            tag = f"Gemini ({model}, {subj})"
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
