"""
modes/anime.py — Anime / Manga Outline (Optimized for clean lines)
==================================================================
Approach: TRUE progressive anime/manga drawing flow
  1. Start with a pure white canvas.
  2. Progressive redraw of structural lineart contours (black ink strokes)
     ordered by length (silhouette first).
  3. Fade in/multiply the flat cel-shaded colors underneath the clean outlines.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw

from src.gpu_utils import (
    gpu_xdog,
    GPU_AVAILABLE,
)


def _thin_edges(binary_img):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(binary_img.shape, np.uint8)
    img = binary_img.copy()
    for _ in range(15):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


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

    # ── Step 1: Cel-shaded color fill base ────────────────────────────────────
    smooth = cv2.bilateralFilter(img_np, d=15, sigmaColor=150, sigmaSpace=150)
    smooth = cv2.bilateralFilter(smooth, d=11, sigmaColor=120, sigmaSpace=120)
    smooth = cv2.medianBlur(smooth, 9)
    cel = _cel_quantise(smooth, levels=3)

    # ── Step 2: XDoG edge detection for clean outlines ───────────────────────
    gray_raw = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray_raw, 11, 80, 80)

    sigma1 = max(0.8, blur_size * 0.18)
    sigma2 = sigma1 * 1.6
    edge_mask = gpu_xdog(gray, sigma1=sigma1, sigma2=sigma2, tau=0.985, phi=16.0, epsilon=-0.05)

    # Clean double outlines and thin to centerline
    edges = 255 - edge_mask
    merged = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    thinned = _thin_edges(merged)
    
    lw = max(1, line_art_width)
    if lw > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lw, lw))
        thinned = cv2.dilate(thinned, kernel, iterations=1)
    edge_mask = 255 - thinned

    # ── Canvas starts as pure white ───────────────────────────────────────────
    pil_white = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    yield pil_white.copy()

    # ── Step 3: Draw outlines progressively ──────────────────────────────────
    cnts, _ = cv2.findContours(255 - edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_sorted = sorted([c for c in cnts if cv2.arcLength(c, False) > 8], 
                         key=lambda c: cv2.arcLength(c, False), reverse=True)

    draw_layer = ImageDraw.Draw(pil_white)
    n = len(cnts_sorted)
    eff_batch = max(1, min(batch_size, max(1, n // 100)))

    for idx, cnt in enumerate(cnts_sorted):
        perim = cv2.arcLength(cnt, False)
        eps = max(0.08, 0.005 * perim)
        approx = cv2.approxPolyDP(cnt, eps, False)
        path = [tuple(pt[0]) for pt in approx]
        if len(path) < 2:
            continue

        if perim > 50:
            area = cv2.contourArea(cnt)
            if area < perim * 1.5:
                path = path[:len(path)//2 + 1]
                if len(path) < 2:
                    continue

        sw = line_art_width if perim > 120 else max(1, line_art_width - 1)
        pts = _jitter(path, jitter * 0.4)
        for i in range(len(pts) - 1):
            draw_layer.line([pts[i], pts[i + 1]], fill=(15, 15, 20, 255), width=sw)

        if idx % eff_batch == 0 or idx == n - 1:
            yield pil_white.copy()

    # Add eye highlights / sparkle
    try:
        _, thresh_eye = cv2.threshold(gray_raw, 222, 255, cv2.THRESH_BINARY)
        cnts_eye, _ = cv2.findContours(thresh_eye, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_eye:
            area = cv2.contourArea(cnt)
            p_len = cv2.arcLength(cnt, True)
            if 2 < area < 200 and p_len < 80:
                approx = cv2.approxPolyDP(cnt, 0.5, True)
                pts = [tuple(pt[0]) for pt in approx]
                if len(pts) >= 3:
                    draw_layer.polygon(pts, fill=(255, 255, 255, 255))
    except Exception as ee:
        print(f"[anime] Catchlight failed: {ee}")

    # ── Step 4: Fade in cel-shaded color layer underneath lineart ─────────────
    lines_rgb = np.array(pil_white.convert("RGB"))
    cel_rgba = Image.fromarray(cel).convert("RGB")
    cel_np = np.array(cel_rgba)

    steps = 15
    for step in range(1, steps + 1):
        alpha = step / float(steps)
        # Fade colors in
        base_color = cv2.addWeighted(np.full((h, w, 3), 255, dtype=np.uint8), 1.0 - alpha, cel_np, alpha, 0)
        # Multiply lines on top
        blended = np.clip((base_color.astype(np.float32) / 255.0) * (lines_rgb.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)
        yield Image.fromarray(blended).convert("RGBA")

    pil_white.close()
