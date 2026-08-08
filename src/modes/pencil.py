"""
modes/pencil.py — Colored Pencil Sketch
========================================
Approach: TRUE progressive colored pencil drawing technique
  1. Start with a pure white paper canvas.
  2. Draw color-sampled outline paths progressively (high contrast local colors).
  3. Draw color-sampled vector shading strokes progressively:
     - Short, wiggled colored pencil lines.
     - Natural human drawing order: top-to-bottom.
     - Saliency-gated.
  4. Fade in a soft, low-contrast color wash underpainting at the end.
  5. Apply linen paper grain texture.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter

from src.gpu_utils import (
    gpu_gaussian_blur,
    gpu_saliency,
    gpu_canvas_texture,
    gpu_multiply,
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


def _generate_color_hatching_strokes(img_rgb, gray_bilateral, saliency_norm, w, h):
    strokes = []
    spacing = max(5, min(w, h) // 100)
    
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            sal = float(saliency_norm[y, x]) if saliency_norm is not None else 1.0
            if sal < 0.08:
                continue
                
            lum = int(gray_bilateral[y, x])
            if lum >= 210:
                continue
                
            c_sampled = img_rgb[y, x].astype(np.float32)
            
            # Midtone hatching (diagonal 45 degrees)
            length = random.uniform(8, 16)
            angle = np.radians(45 + random.uniform(-12, 12))
            dx = np.cos(angle) * length
            dy = np.sin(angle) * length
            
            x1 = x - dx/2 + random.uniform(-1, 1)
            y1 = y - dy/2 + random.uniform(-1, 1)
            x2 = x + dx/2 + random.uniform(-1, 1)
            y2 = y + dy/2 + random.uniform(-1, 1)
            
            stroke_factor = 0.35 + 0.25 * (lum / 255.0)
            stroke_rgb = tuple(max(10, int(c * stroke_factor)) for c in c_sampled)
            
            strokes.append(((x1, y1), (x2, y2), stroke_rgb, 1))
            
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
                
                stroke_factor_c = 0.20 + 0.15 * (lum / 255.0)
                stroke_rgb_c = tuple(max(5, int(c * stroke_factor_c)) for c in c_sampled)
                
                strokes.append(((cx1, cy1), (cx2, cy2), stroke_rgb_c, 1))
                
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
    threshold_block: int = 11,
    threshold_c: int = 4,
    jitter: float = 0.40,
    batch_size: int = 10,
    **_kw,
):
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))
    gray_raw = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    gray = cv2.bilateralFilter(gray_raw, 7, 30, 30)
    saliency_norm = gpu_saliency(gray_raw)

    # ── Canvas starts as pure white paper ─────────────────────────────────────
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    yield canvas.copy()

    # ── Step 1: Draw outline paths progressively ──────────────────────────────
    clo = max(15, 60 - threshold_c * 5)
    chi = max(45, 140 - threshold_c * 8)
    paths = _build_contour_paths(gray, clo, chi, min_path_len=8)

    paths.sort(key=lambda p: _path_saliency(p, saliency_norm, w, h), reverse=True)

    draw_layer = ImageDraw.Draw(canvas)
    n_out = len(paths)
    eff_out = max(1, min(batch_size, max(1, n_out // 120)))

    for idx, path in enumerate(paths):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        c_sampled = img_np[sy, sx].astype(np.float32)
        lum = float(gray[sy, sx])
        path_sal = _path_saliency(path, saliency_norm, w, h)

        if path_sal < 0.10:
            if len(path) < 12:
                continue
            stroke_factor = 0.80 + random.uniform(0, 0.15)
            line_w = 1
        elif path_sal < 0.28:
            stroke_factor = 0.58 + random.uniform(0, 0.10)
            line_w = 1
        else:
            if lum < 40:
                stroke_factor = 0.07 + random.uniform(0, 0.04)
                line_w = 3
            elif lum < 80:
                stroke_factor = 0.11 + random.uniform(0, 0.05)
                line_w = 2
            elif lum < 140:
                stroke_factor = 0.20 + random.uniform(0, 0.07)
                line_w = 2
            elif lum < 190:
                stroke_factor = 0.35 + random.uniform(0, 0.08)
                line_w = 2
            else:
                stroke_factor = 0.52 + random.uniform(0, 0.10)
                line_w = 1

        stroke_rgb = tuple(max(5, int(c * stroke_factor)) for c in c_sampled)

        pts = _jitter(path, jitter)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                draw_layer.line([pts[i], pts[i + 1]], fill=(*stroke_rgb, 255), width=line_w)

        if idx % eff_out == 0 or idx == n_out - 1:
            yield canvas.copy()

    # ── Step 2: Draw color hatching strokes progressively ────────────────────
    hatching_strokes = _generate_color_hatching_strokes(img_np, gray, saliency_norm, w, h)
    n_hatch = len(hatching_strokes)
    eff_hatch = max(5, min(batch_size * 5, max(5, n_hatch // 100)))

    for idx, (pt1, pt2, stroke_rgb, lw) in enumerate(hatching_strokes):
        x1, y1 = pt1
        x2, y2 = pt2
        mx = (x1 + x2) / 2 + random.uniform(-0.6, 0.6)
        my = (y1 + y2) / 2 + random.uniform(-0.6, 0.6)
        
        draw_layer.line([pt1, (mx, my), pt2], fill=(*stroke_rgb, 255), width=lw)
        
        if idx % eff_hatch == 0 or idx == n_hatch - 1:
            yield canvas.copy()

    # ── Step 3: Fade in color wash base under outlines at the end ─────────────
    lines_np = np.array(canvas.convert("RGB"))
    
    blur_r = max(20, min(w, h) // 5)
    washed = np.array(
        pil_img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_r))
    ).astype(np.float32)
    wash_blended = np.clip(255.0 * 0.60 + washed * 0.40, 0, 255).astype(np.uint8)

    steps = 15
    for step in range(1, steps + 1):
        alpha = step / float(steps)
        base_color = cv2.addWeighted(np.full((h, w, 3), 255, dtype=np.uint8), 1.0 - alpha, wash_blended, alpha, 0)
        blended = np.clip((base_color.astype(np.float32) / 255.0) * (lines_np.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)
        yield Image.fromarray(blended).convert("RGBA")

    # Paper texture
    try:
        final_canvas = Image.fromarray(blended).convert("RGBA")
        canvas_tex = gpu_canvas_texture(w, h)
        canvas_rgb = np.array(final_canvas.convert("RGB")).astype(np.float32) / 255.0
        tex_f = canvas_tex.astype(np.float32)[:, :, np.newaxis] / 255.0
        multiplied = canvas_rgb * np.repeat(tex_f, 3, axis=2)
        blended_tex = np.clip(canvas_rgb * 0.92 + multiplied * 0.08, 0, 1)
        final_canvas = Image.fromarray((blended_tex * 255).astype(np.uint8)).convert("RGBA")
        yield final_canvas.copy()
        final_canvas.close()
    except Exception as te:
        print(f"[pencil] Paper texture blend failed: {te}")
        yield Image.fromarray(blended).convert("RGBA")

    canvas.close()
