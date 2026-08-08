"""
modes/sketch.py — Realistic Charcoal/Pencil Sketch
=====================================================
Approach: TRUE progressive artist sketch technique
  1. Start with a pure white paper canvas.
  2. Draw the outline contours first (silhouette & structure) from center-out/outside-in.
  3. Draw detail lines (features, cloth folds) with fine graphite strokes.
  4. Fade in the value shading layer (shading, core shadows) progressively *after* outlines
     are drawn, simulating hand-smudged charcoal or soft graphite shading.
  5. Saliency gating ensures background walls stay clean white.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw

from src.gpu_utils import (
    gpu_gaussian_blur,
    gpu_saliency,
    GPU_AVAILABLE,
)


def _jitter(pts, amount: float):
    result, dx, dy = [], 0.0, 0.0
    for x, y in pts:
        dx = 0.82 * dx + 0.18 * random.gauss(0, amount * 1.8)
        dy = 0.82 * dy + 0.18 * random.gauss(0, amount * 1.8)
        result.append((x + dx, y + dy))
    return result


def _thin_edges(binary_img):
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(binary_img.shape, np.uint8)
    img = binary_img.copy()
    for _ in range(12):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _build_contour_paths(gray_bilateral, clo, chi, min_path_len=8):
    edges = cv2.Canny(gray_bilateral, clo, chi)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k)
    merge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    merged = cv2.dilate(edges, merge_k, iterations=1)
    thinned = _thin_edges(merged)
    cnts, _ = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in cnts:
        p = cv2.arcLength(cnt, False)
        if p < min_path_len:
            continue
        eps = max(0.3, 0.00015 * p)  # Higher detail resolution
        approx = cv2.approxPolyDP(cnt, eps, False)
        if len(approx) > 1:
            path = [tuple(pt[0]) for pt in approx]
            if len(path) > 1:
                paths.append(path)
    return paths


def _build_value_shading(gray_raw, saliency_norm):
    h, w = gray_raw.shape
    shading = np.full((h, w), 255.0, dtype=np.float32)

    smooth = cv2.GaussianBlur(gray_raw.astype(np.float32), (25, 25), 0)
    lum = smooth

    if saliency_norm is not None:
        sal = np.clip((saliency_norm - 0.08) / 0.35, 0, 1)
    else:
        sal = np.ones((h, w), dtype=np.float32)

    # Midtone
    mid_strength = np.clip((220.0 - lum) / (220.0 - 80.0), 0, 1) * sal
    shading = shading - mid_strength * (255.0 - 155.0)

    # Shadow
    shadow_strength = np.clip((160.0 - lum) / 160.0, 0, 1) * sal
    shading = shading - shadow_strength * (shading - 15.0)

    return np.clip(shading, 0, 255).astype(np.uint8)


def _path_saliency(path, saliency_norm, w, h):
    if saliency_norm is None:
        return 1.0
    vals = []
    step = max(1, len(path) // 8)
    for pt in path[::step]:
        x = max(0, min(w - 1, int(pt[0])))
        y = max(0, min(h - 1, int(pt[1])))
        vals.append(float(saliency_norm[y, x]))
    return float(np.mean(vals)) if vals else 0.0


def draw(
    pil_img: Image.Image,
    *,
    blur_size: int = 3,
    threshold_c: int = 5,
    jitter: float = 0.45,
    batch_size: int = 10,
    **_kw,
):
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))
    gray_raw = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    gray = cv2.bilateralFilter(gray_raw, 7, 30, 30)
    saliency_norm = gpu_saliency(gray_raw)

    # ── Canvas starts as pure white paper ─────────────────────────────────────
    canvas_np = np.full((h, w, 3), 255, dtype=np.uint8)
    pil_canvas = Image.fromarray(canvas_np)
    yield pil_canvas.copy()

    # ── Step 1: Extract and draw outlines first (No pre-pasted shading) ──────
    clo = max(15, 60 - threshold_c * 5)
    chi = max(45, 140 - threshold_c * 8)
    outlines = _build_contour_paths(gray, clo, chi, min_path_len=8)

    cx0, cy0 = w / 2.0, h / 2.0
    # Prioritize center subject contours drawing first
    outlines.sort(key=lambda p: _path_saliency(p, saliency_norm, w, h), reverse=True)

    draw_layer = ImageDraw.Draw(pil_canvas)

    n_out = len(outlines)
    # Scale yield speed to increase frame counts smoothly
    eff_out = max(1, min(batch_size, max(1, n_out // 120)))

    for idx, path in enumerate(outlines):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        lum = int(gray[sy, sx])
        path_sal = _path_saliency(path, saliency_norm, w, h)

        if path_sal < 0.10:
            if len(path) < 12:
                continue
            tone = random.randint(180, 220)
            line_w = 1
        elif path_sal < 0.28:
            tone = random.randint(120, 160)
            line_w = 1
        else:
            if lum < 40:
                tone = random.randint(3, 15)
                line_w = 3
            elif lum < 90:
                tone = random.randint(8, 25)
                line_w = 2
            elif lum < 150:
                tone = random.randint(20, 50)
                line_w = 2
            else:
                tone = random.randint(50, 95)
                line_w = 1

        pts = _jitter(path, jitter)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                draw_layer.line([pts[i], pts[i + 1]], fill=(tone, tone, tone), width=line_w)

        if idx % eff_out == 0 or idx == n_out - 1:
            yield pil_canvas.copy()

    # ── Step 2: Progressively blend the hand-drawn shading (saliency-gated) ───
    shading_gray = _build_value_shading(gray_raw, saliency_norm)
    shading_rgb = cv2.cvtColor(shading_gray, cv2.COLOR_GRAY2RGB)

    lines_np = np.array(pil_canvas.convert("RGB"))

    # Transition from pure outlines to shaded drawing in 15 progressive frames
    steps = 15
    for step in range(1, steps + 1):
        alpha = step / float(steps)
        # Composite shading underneath outline lines
        shaded_base = cv2.addWeighted(np.full((h, w, 3), 255, dtype=np.uint8), 1.0 - alpha, shading_rgb, alpha, 0)
        # Multiply outlines on top
        blended = np.clip((shaded_base.astype(np.float32) / 255.0) * (lines_np.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)
        yield Image.fromarray(blended)

    pil_canvas.close()
