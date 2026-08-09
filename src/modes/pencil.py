"""
modes/pencil.py — Colored Pencil Sketch Drawing
=================================================
GPU Acceleration via src.gpu_utils (CuPy → CPU fallback):
  - Pencil dodge blend      → gpu_pencil_dodge_channel (GPU)
  - Color multiply composite→ gpu_multiply (GPU)
  - Gaussian blur           → gpu_gaussian_blur (GPU)
  - Canny edge paths        → gpu_canny (blur on GPU, Canny on CPU)
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter

from src.gpu_utils import (
    gpu_gaussian_blur,
    gpu_pencil_dodge_channel,
    gpu_multiply,
    gpu_canny,
    GPU_AVAILABLE,
)


def _jitter(pts, amount: float):
    result, dx, dy = [], 0.0, 0.0
    for x, y in pts:
        dx = 0.82 * dx + 0.18 * random.gauss(0, amount * 1.5)
        dy = 0.82 * dy + 0.18 * random.gauss(0, amount * 1.5)
        result.append((x + dx, y + dy))
    return result


def _hatch_shadow(gray, spacing: int, dark_thresh: int, angle: int):
    h, w = gray.shape
    lines = []
    for offset in range(-w, h, spacing):
        current, in_dark = [], False
        for x in range(w):
            y = x + offset
            if 0 <= y < h:
                if gray[y, x] < dark_thresh:
                    if not in_dark:
                        in_dark = True
                        current = [(x, y)]
                    else:
                        current.append((x, y))
                else:
                    if in_dark and len(current) > 5:
                        lines.append(current)
                    in_dark = False
                    current = []
            else:
                if in_dark and len(current) > 5:
                    lines.append(current)
                in_dark = False
                current = []
        if in_dark and len(current) > 5:
            lines.append(current)
    return lines


def _combined_paths(gray, blur, clo, chi, block, c_val, eps=0.0004):
    b = max(3, blur | 1)
    # GPU-accelerated blur, then CPU Canny + AdaptiveThreshold
    blurred = gpu_gaussian_blur(gray, b)
    edges   = cv2.Canny(blurred, clo, chi)
    bk = max(3, block | 1)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, bk, c_val + 2
    )
    combined = cv2.bitwise_or(edges, thresh)
    cnts, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in cnts:
        p = cv2.arcLength(cnt, False)
        if p < 3:
            continue
        approx = cv2.approxPolyDP(cnt, max(0.04, eps * p), False)
        if len(approx) > 1:
            paths.append([tuple(pt[0]) for pt in approx])
    return paths


def draw(
    pil_img: Image.Image,
    *,
    blur_size: int = 3,
    threshold_block: int = 11,
    threshold_c: int = 4,
    jitter: float = 0.35,
    bg_color_wash: bool = True,
    wash_opacity: int = 75,
    sketch_opacity: float = 0.13,
    batch_size: int = 10,
    **_kw,
):
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))
    gray_raw = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Bilateral filtering to clear JPEG noise blocks in dark shadows
    gray = cv2.bilateralFilter(gray_raw, 7, 30, 30)

    # ── Layer 1: Colour wash underpainting ───────────────────────────────────
    canvas = Image.new("RGBA", (w, h), (255, 252, 244, 255))

    blur_r = max(14, min(w, h) // 8)
    washed_pil  = pil_img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_r))
    from PIL import ImageEnhance
    # Boost color saturation by 1.6x to make colored pencil shades pop vibrantly
    washed_pil = ImageEnhance.Color(washed_pil).enhance(1.6)
    washed_rgba = washed_pil.convert("RGBA")
    washed_pil.close()

    opacity = min(220, max(60, wash_opacity + 40))
    r_ch, g_ch, b_ch, _ = washed_rgba.split()
    a_ch = Image.new("L", (w, h), opacity)
    washed_rgba.putalpha(a_ch)
    canvas.alpha_composite(washed_rgba)
    washed_rgba.close()

    yield canvas.copy()

    # ── Layer 2: Per-channel pencil dodge (GPU × 3) ───────────────────────────
    rgb_np = np.array(canvas.convert("RGB"))

    pencil_r = gpu_pencil_dodge_channel(img_np[:, :, 0], blur_size)  # GPU
    pencil_g = gpu_pencil_dodge_channel(img_np[:, :, 1], blur_size)  # GPU
    pencil_b = gpu_pencil_dodge_channel(img_np[:, :, 2], blur_size)  # GPU
    pencil_rgb = np.stack([pencil_r, pencil_g, pencil_b], axis=2).astype(np.float32) / 255.0

    # GPU multiply blend
    base_f  = rgb_np.astype(np.float32) / 255.0
    blend_f = gpu_multiply(base_f, pencil_rgb)                        # GPU
    canvas  = Image.fromarray((blend_f * 255).astype(np.uint8)).convert("RGBA")
    yield canvas.copy()

    # ── Layer 3: Crosshatch in shadow zones ──────────────────────────────────
    spacing     = max(8, min(w, h) // 65)
    hatch_lines = _hatch_shadow(gray, spacing=spacing, dark_thresh=75, angle=45)

    draw_layer = ImageDraw.Draw(canvas)
    for idx, hatch in enumerate(hatch_lines):
        if not hatch:
            continue
        sx = max(0, min(w - 1, int(hatch[0][0])))
        sy = max(0, min(h - 1, int(hatch[0][1])))
        c_sampled = tuple(img_np[sy, sx])
        
        # Calculate local luminance to determine shading factor dynamically
        lum = int(gray[sy, sx])
        # Dark areas get deeper colors (65%-75%), light areas get softer tints (85%-95%)
        shade_factor = 0.65 + 0.30 * (lum / 255.0)
        shade = tuple(max(0, int(c * shade_factor)) for c in c_sampled)
        
        pts   = _jitter(hatch, jitter * 0.5)
        if len(pts) > 1:
            draw_layer.line(pts, fill=(*shade, 150), width=1)
        if idx % (batch_size * 2) == 0 or idx == len(hatch_lines) - 1:
            yield canvas.copy()

    # ── Layer 4: Colour contour strokes ──────────────────────────────────────
    paths = _combined_paths(gray, blur_size, 35, 110, threshold_block, threshold_c, eps=0.0004)

    # Highlight (Sun) detection in pencil sketch
    try:
        _, thresh_sun = cv2.threshold(gray_raw, 225, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh_sun = cv2.morphologyEx(thresh_sun, cv2.MORPH_OPEN, kernel)
        cnts_sun, _ = cv2.findContours(thresh_sun, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_sun:
            p_len = cv2.arcLength(cnt, True)
            if 30 < p_len < (w + h) * 0.5:
                approx = cv2.approxPolyDP(cnt, max(1.0, 0.003 * p_len), True)
                if len(approx) > 4:
                    paths.append([tuple(pt[0]) for pt in approx])
    except Exception as se:
        print(f"[pencil] Highlight extraction error: {se}")

    cx0, cy0 = w / 2.0, h / 2.0
    paths.sort(key=lambda p: -((p[0][0] - cx0)**2 + (p[0][1] - cy0)**2)**0.5)

    n = len(paths)
    eff_batch = max(1, min(batch_size, n // 300)) if n > 0 else batch_size

    for idx, path in enumerate(paths):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        c_sampled = tuple(img_np[sy, sx])
        
        # Calculate local luminance for contour stroke weight
        lum = int(gray[sy, sx])
        stroke_factor = 0.55 + 0.40 * (lum / 255.0)
        stroke = tuple(max(0, int(c * stroke_factor)) for c in c_sampled)
        
        pts    = _jitter(path, jitter)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                draw_layer.line([pts[i], pts[i + 1]], fill=(*stroke, 200), width=1)
        if idx % eff_batch == 0 or idx == len(paths) - 1:
            yield canvas.copy()

    # Blend a subtle linen texture into the colored pencil sketch to simulate paper grain
    from src.gpu_utils import gpu_canvas_texture
    try:
        canvas_tex = gpu_canvas_texture(w, h)
        canvas_rgb = np.array(canvas.convert("RGB"))
        canvas_f = canvas_rgb.astype(np.float32) / 255.0
        tex_f = canvas_tex.astype(np.float32)[:, :, np.newaxis] / 255.0
        # Light blend: mix 85% of standard canvas with 15% multiplied canvas texture
        multiplied = gpu_multiply(canvas_f, np.repeat(tex_f, 3, axis=2))
        blended = cv2.addWeighted(canvas_f, 0.85, multiplied, 0.15, 0)
        canvas = Image.fromarray((np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGBA")
    except Exception as te:
        print(f"[pencil] Linen texture blending failed: {te}")

    yield canvas.copy()
