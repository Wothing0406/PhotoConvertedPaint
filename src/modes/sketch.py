"""
modes/sketch.py — Realistic Sketch / Charcoal Drawing
=======================================================
GPU Acceleration via src.gpu_utils (CuPy → CPU fallback):
  - Pencil dodge blend      → gpu_pencil_dodge     (GPU)
  - Dark shadow composite   → gpu_dark_mask_composite (GPU)
  - Gaussian blur           → gpu_gaussian_blur    (GPU)
  - Canny edge paths        → gpu_canny            (blur on GPU, Canny on CPU)
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter

from src.gpu_utils import (
    gpu_gaussian_blur,
    gpu_pencil_dodge,
    gpu_dark_mask_composite,
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


def _canny_paths(gray, lo, hi, blur, eps_factor=0.0005):
    edges = gpu_canny(gray, lo, hi, blur)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in cnts:
        p = cv2.arcLength(cnt, False)
        if p < 3:
            continue
        approx = cv2.approxPolyDP(cnt, max(0.05, eps_factor * p), False)
        if len(approx) > 1:
            paths.append([tuple(pt[0]) for pt in approx])
    return paths


def draw(
    pil_img: Image.Image,
    *,
    blur_size: int = 3,
    threshold_c: int = 5,
    jitter: float = 0.40,
    bg_color_wash: bool = True,
    wash_opacity: int = 50,
    batch_size: int = 10,
    **_kw,
):
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # ── Layer 1: Warm paper background ───────────────────────────────────────
    paper = np.full((h, w, 3), (245, 240, 228), dtype=np.uint8)

    if bg_color_wash and wash_opacity > 0:
        blur_r = max(20, min(w, h) // 7)
        washed = np.array(
            pil_img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_r))
        )
        g3 = cv2.cvtColor(cv2.cvtColor(washed, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        washed = cv2.addWeighted(washed, 0.20, g3, 0.80, 0)
        alpha = min(0.50, wash_opacity / 180.0)
        paper = cv2.addWeighted(paper, 1.0 - alpha, washed, alpha, 0)

    canvas_np = paper.copy()
    pil_canvas = Image.fromarray(canvas_np)
    yield pil_canvas.copy()
    pil_canvas.close()

    # ── Layer 2: Pencil dodge-blend texture (GPU) ─────────────────────────────
    pencil_tex = gpu_pencil_dodge(gray, blur_size)          # GPU accelerated

    canvas_f = canvas_np.astype(np.float32) / 255.0
    pencil_f = pencil_tex.astype(np.float32)[:, :, np.newaxis] / 255.0   # (H,W,1)
    blended  = np.clip(canvas_f * pencil_f, 0.0, 1.0)
    canvas_np = (blended * 255).astype(np.uint8)

    pil_canvas = Image.fromarray(canvas_np)
    yield pil_canvas.copy()
    pil_canvas.close()

    # ── Layer 3: Tonal shadow darkening (GPU) ────────────────────────────────
    dark_raw = (gray < 85).astype(np.uint8) * 255
    dark_mask = gpu_gaussian_blur(dark_raw, 19).astype(np.float32) / 255.0   # GPU

    shadow_strength = _kw.get("shadow_strength", 0.35)
    canvas_np = gpu_dark_mask_composite(canvas_np.astype(np.float32), dark_mask, strength=shadow_strength)

    pil_canvas = Image.fromarray(canvas_np)
    yield pil_canvas.copy()
    pil_canvas.close()

    # ── Layer 4: Canny structural edge strokes ────────────────────────────────
    clo = max(15, 65 - threshold_c * 6)
    chi = max(60, 150 - threshold_c * 9)
    paths = _canny_paths(gray, clo, chi, blur_size, eps_factor=0.0005)

    cx0, cy0 = w / 2.0, h / 2.0
    paths.sort(key=lambda p: -((p[0][0] - cx0)**2 + (p[0][1] - cy0)**2)**0.5)

    pil_canvas = Image.fromarray(canvas_np)
    draw_layer = ImageDraw.Draw(pil_canvas)

    n = len(paths)
    eff_batch = max(1, min(batch_size, n // 300)) if n > 0 else batch_size

    for idx, path in enumerate(paths):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        lum  = int(gray[sy, sx])
        tone = max(6, min(140, int(lum * 0.48)))
        pts  = _jitter(path, jitter)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                draw_layer.line([pts[i], pts[i + 1]], fill=(tone, tone, tone), width=1)
        if idx % eff_batch == 0 or idx == n - 1:
            yield pil_canvas.copy()

    yield pil_canvas.copy()
    pil_canvas.close()
