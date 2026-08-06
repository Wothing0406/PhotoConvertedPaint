"""
modes/anime.py — Anime / Manga Outline (Optimized for clean lines)
==================================================================
Fix: To prevent massive black lines everywhere on complex backgrounds:
  - We use standard Bilateral Filter (which preserves strong anime outlines).
  - We simplify XDoG parameters so it only detects primary contours, ignoring tiny texture noise.
  - Adaptive threshold + morphological cleanup to ensure flat cel shading.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw

from src.gpu_utils import (
    gpu_xdog,
    GPU_AVAILABLE,
)


def _jitter(pts, amount: float):
    result, dx, dy = [], 0.0, 0.0
    for x, y in pts:
        dx = 0.80 * dx + 0.20 * random.gauss(0, amount * 1.2)
        dy = 0.80 * dy + 0.20 * random.gauss(0, amount * 1.2)
        result.append((x + dx, y + dy))
    return result


def _cel_quantise(img_np: np.ndarray, levels: int = 4) -> np.ndarray:
    step = 256 // levels
    q = (img_np.astype(np.float32) / step).astype(np.uint8) * step
    return np.clip(q + step // 2, 0, 255).astype(np.uint8)


def draw(
    pil_img: Image.Image,
    *,
    blur_size: int = 5,
    threshold_c: int = 10,
    jitter: float = 0.15,
    line_art_width: int = 2,
    batch_size: int = 10,
    **_kw,
):
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # ── Step 1: Smooth for color fill (Bilateral filters on CPU to preserve cartoon borders) ──
    # Bilateral smoothing makes colors extremely flat like a real cel-shaded anime frame
    smooth = cv2.bilateralFilter(img_np, d=15, sigmaColor=150, sigmaSpace=150)
    smooth = cv2.bilateralFilter(smooth, d=11, sigmaColor=120, sigmaSpace=120)
    smooth = cv2.medianBlur(smooth, 9)

    # ── Step 2: Cel-shade quantization ───────────────────────────────────────
    cel = _cel_quantise(smooth, levels=3)

    # ── Step 3: Clean XDoG edge map (optimized parameters for clean lines) ───
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # We use larger sigmas to ignore small swirling details and only catch major boundaries
    sigma1 = max(0.8, blur_size * 0.18)
    sigma2 = sigma1 * 1.6
    
    # Tuned epsilon and tau to prune fine details (like Van Gogh starry night swirls)
    edge_mask = gpu_xdog(gray, sigma1=sigma1, sigma2=sigma2,
                         tau=0.99, phi=15.0, epsilon=-0.02)

    # Median blur on edges to remove salt-and-pepper noise spots
    edge_mask = cv2.medianBlur(edge_mask, 3)

    lw = max(1, line_art_width)
    if lw > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lw, lw))
        edge_mask = cv2.erode(edge_mask, kernel, iterations=1)

    # ── Step 4: Build canvas ─────────────────────────────────────────────────
    canvas_np = cel.copy()
    ink_mask = (edge_mask == 0)
    canvas_np[ink_mask] = (15, 15, 20)  # Clean dark anime line color

    canvas = Image.fromarray(canvas_np)
    yield canvas.copy()

    # ── Step 5: Progressive redraw of structural contours ────────────────────
    cnts, _ = cv2.findContours(255 - edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Only draw long clean lines
    cnts_sorted = sorted([c for c in cnts if cv2.arcLength(c, False) > 20], 
                         key=lambda c: cv2.arcLength(c, False), reverse=True)

    pil_white = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    pil_white.paste(Image.fromarray(cel), (0, 0))
    draw_layer = ImageDraw.Draw(pil_white)

    n = len(cnts_sorted)
    eff_batch = max(1, min(batch_size, n // 200)) if n > 0 else batch_size

    for idx, cnt in enumerate(cnts_sorted):
        perim = cv2.arcLength(cnt, False)
        eps = max(0.08, 0.005 * perim)  # Simpler lines (less nodes)
        approx = cv2.approxPolyDP(cnt, eps, False)
        path = [tuple(pt[0]) for pt in approx]
        if len(path) < 2:
            continue

        sw = line_art_width if perim > 120 else max(1, line_art_width - 1)
        pts = _jitter(path, jitter * 0.4)
        for i in range(len(pts) - 1):
            draw_layer.line([pts[i], pts[i + 1]], fill=(15, 15, 20, 255), width=sw)

        if idx % eff_batch == 0 or idx == n - 1:
            yield pil_white.copy()

    yield pil_white.copy()
