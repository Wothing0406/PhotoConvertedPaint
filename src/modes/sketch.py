"""
modes/sketch.py — Realistic Charcoal/Pencil Sketch
=====================================================
Approach: TRUE progressive vector-hatching sketch technique
  1. Start with a pure white paper canvas.
  2. Draw structural outline contours progressively (outside-in / center-out).
  3. Draw vector-hatching shading strokes progressively:
     - Short, organic, wiggled pencil strokes (length 8-16px).
     - No uniform grid: sampled on random grid with angular and positional jitter.
     - Natural human drawing order: drawn from top-to-bottom to simulate hand shading.
     - Saliency-gated to keep background pure white.
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
        eps = max(0.3, 0.00015 * p)
        approx = cv2.approxPolyDP(cnt, eps, False)
        if len(approx) > 1:
            path = [tuple(pt[0]) for pt in approx]
            if len(path) > 1:
                paths.append(path)
    return paths


def _generate_hatching_strokes(gray_bilateral, saliency_norm, w, h):
    strokes = []
    # Dynamic spacing based on image size to balance speed and quality
    spacing = max(5, min(w, h) // 100)
    
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            sal = float(saliency_norm[y, x]) if saliency_norm is not None else 1.0
            if sal < 0.08:
                continue
                
            lum = int(gray_bilateral[y, x])
            if lum >= 210:
                continue  # Highlight: leave paper white
                
            # Midtone hatching (diagonal 45 degrees)
            length = random.uniform(8, 16)
            angle = np.radians(45 + random.uniform(-12, 12))
            dx = np.cos(angle) * length
            dy = np.sin(angle) * length
            
            x1 = x - dx/2 + random.uniform(-1, 1)
            y1 = y - dy/2 + random.uniform(-1, 1)
            x2 = x + dx/2 + random.uniform(-1, 1)
            y2 = y + dy/2 + random.uniform(-1, 1)
            
            tone = random.randint(80, 130) if lum < 120 else random.randint(130, 180)
            strokes.append(((x1, y1), (x2, y2), tone, 1))
            
            # Deep shadow cross-hatching (diagonal 135 degrees)
            if lum < 120:
                length_c = random.uniform(6, 12)
                angle_c = np.radians(135 + random.uniform(-12, 12))
                dx_c = np.cos(angle_c) * length_c
                dy_c = np.sin(angle_c) * length_c
                
                cx1 = x - dx_c/2 + random.uniform(-1, 1)
                cy1 = y - dy_c/2 + random.uniform(-1, 1)
                cx2 = x + dx_c/2 + random.uniform(-1, 1)
                cy2 = y + dy_c/2 + random.uniform(-1, 1)
                
                tone_c = random.randint(30, 80)
                strokes.append(((cx1, cy1), (cx2, cy2), tone_c, 1))
                
    # Sort strokes from top-to-bottom with horizontal jitter to simulate human hand drawing
    strokes.sort(key=lambda s: s[0][1] + random.uniform(-40, 40))
    return strokes


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

    # ── Step 1: Draw outlines progressively ───────────────────────────────────
    clo = max(15, 60 - threshold_c * 5)
    chi = max(45, 140 - threshold_c * 8)
    outlines = _build_contour_paths(gray, clo, chi, min_path_len=8)

    # Draw primary features/subject first
    outlines.sort(key=lambda p: _path_saliency(p, saliency_norm, w, h), reverse=True)

    draw_layer = ImageDraw.Draw(pil_canvas)
    n_out = len(outlines)
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

    # ── Step 2: Draw organic vector-hatching strokes progressively ────────────
    hatching_strokes = _generate_hatching_strokes(gray, saliency_norm, w, h)
    n_hatch = len(hatching_strokes)
    # Calibrate yielding speed to show drawing process beautifully
    eff_hatch = max(5, min(batch_size * 5, max(5, n_hatch // 100)))

    for idx, (pt1, pt2, tone, lw) in enumerate(hatching_strokes):
        # Add a tiny wiggle to shading lines to make them look hand-drawn
        x1, y1 = pt1
        x2, y2 = pt2
        mx = (x1 + x2) / 2 + random.uniform(-0.6, 0.6)
        my = (y1 + y2) / 2 + random.uniform(-0.6, 0.6)
        
        draw_layer.line([pt1, (mx, my), pt2], fill=(tone, tone, tone), width=lw)
        
        if idx % eff_hatch == 0 or idx == n_hatch - 1:
            yield pil_canvas.copy()

    yield pil_canvas.copy()
    pil_canvas.close()
