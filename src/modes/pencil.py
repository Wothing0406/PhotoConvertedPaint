"""
modes/pencil.py — Colored Pencil Sketch
========================================
Approach: TRUE progressive colored pencil technique
  1. Start with a pure white paper canvas.
  2. Draw color-sampled outline paths progressively (high contrast local colors).
  3. Draw detail paths progressively.
  4. Smoothly fade in the color value shading layer + soft underpainting wash
     underneath the drawn outlines over 15 progressive frames.
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


def _build_color_value_shading(img_rgb, gray_raw, saliency_norm):
    h, w = gray_raw.shape
    blur_r = max(15, min(w, h) // 8)
    blur_k = blur_r * 2 + 1
    color_blurred = cv2.GaussianBlur(img_rgb, (blur_k, blur_k), 0).astype(np.float32)
    smooth_lum = cv2.GaussianBlur(gray_raw.astype(np.float32), (25, 25), 0)

    if saliency_norm is not None:
        sal = np.clip((saliency_norm - 0.08) / 0.35, 0, 1)
    else:
        sal = np.ones((h, w), dtype=np.float32)

    result = np.full((h, w, 3), 255.0, dtype=np.float32)

    # Midtone
    mid_strength = np.clip((220.0 - smooth_lum) / (220.0 - 80.0), 0, 1) * sal
    mid_color = np.clip(color_blurred * 0.70 + 80.0, 0, 255)
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1.0 - mid_strength * 0.60)
            + mid_color[:, :, c] * (mid_strength * 0.60)
        )

    # Shadow
    shadow_strength = np.clip((160.0 - smooth_lum) / 160.0, 0, 1) * sal
    dark_color = np.clip(color_blurred * 0.22, 0, 255)
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1.0 - shadow_strength)
            + dark_color[:, :, c] * shadow_strength
        )

    return np.clip(result, 0, 255).astype(np.uint8)


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

    # ── Step 1: Extract and draw outline paths progressively ──────────────────
    clo = max(15, 60 - threshold_c * 5)
    chi = max(45, 140 - threshold_c * 8)
    paths = _build_contour_paths(gray, clo, chi, min_path_len=8)

    cx0, cy0 = w / 2.0, h / 2.0
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

    # ── Step 2: Build underpainting & color value shading layers ──────────────
    # Underpainting wash base
    blur_r = max(20, min(w, h) // 5)
    washed = np.array(
        pil_img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_r))
    ).astype(np.float32)
    wash_blended = np.clip(255.0 * 0.60 + washed * 0.40, 0, 255)

    # Shading overlay
    shading_rgb = _build_color_value_shading(img_np, gray_raw, saliency_norm).astype(np.float32)

    # Composite wash + shading
    combined_color = np.clip(wash_blended * 0.30 + shading_rgb * 0.70, 0, 255).astype(np.uint8)

    # ── Step 3: Progressively blend colors under outlines (15 frames) ─────────
    lines_np = np.array(canvas.convert("RGB"))

    steps = 15
    for step in range(1, steps + 1):
        alpha = step / float(steps)
        # Fade colors in
        base_color = cv2.addWeighted(np.full((h, w, 3), 255, dtype=np.uint8), 1.0 - alpha, combined_color, alpha, 0)
        # Multiply lines on top
        blended = np.clip((base_color.astype(np.float32) / 255.0) * (lines_np.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)
        yield Image.fromarray(blended).convert("RGBA")

    # Final step: paper texture
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
