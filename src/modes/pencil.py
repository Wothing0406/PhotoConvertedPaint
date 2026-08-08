"""
modes/pencil.py — Colored Pencil Sketch
========================================
Approach: MASTER progressive colored pencil technique
  1. Start with a pure white paper canvas.
  2. Single-line outline extraction:
     - Hessian Ridge Detection + XDoG/Canny boundaries.
     - Thin borders with a 9x9 dilation merge to completely avoid double outlines.
     - Smooth coordinates using moving average.
     - No jitter on subject details (sal > 0.20) to prevent distortion.
  3. Cross-Contour Hatching:
     - Short, wiggled color-sampled strokes oriented along the local contour tangents.
     - Deep shadows get saturated dark colored strokes (low stroke factor).
     - Skin highlights (sal > 0.45, lum > 140) remain clean white.
  4. Eye Catchlight Protection:
     - Highlight spots from original image remain pure white.
  5. Fade in soft color wash underpainting & blend linen paper texture.
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
        dx = 0.82 * dx + 0.18 * random.gauss(0, amount * 1.5)
        dy = 0.82 * dy + 0.18 * random.gauss(0, amount * 1.5)
        result.append((x + dx, y + dy))
    return result


def _smooth_path(path, window_size=3):
    if len(path) < window_size:
        return path
    smoothed = []
    for i in range(len(path)):
        start = max(0, i - window_size // 2)
        end = min(len(path), i + window_size // 2 + 1)
        window = path[start:end]
        xs = [pt[0] for pt in window]
        ys = [pt[1] for pt in window]
        smoothed.append((sum(xs)/len(window), sum(ys)/len(window)))
    return smoothed


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


def _build_contour_paths(gray, saliency_norm, clo, chi, min_path_len=8):
    h, w = gray.shape

    # 1. Hessian Ridge
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.8)
    Ixx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
    val = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2))
    
    ridges = np.zeros_like(gray, dtype=np.uint8)
    ridges[val > 5.0] = 255

    # 2. Canny Boundaries
    edges = cv2.Canny(gray, clo, chi)
    
    # 9x9 dilation merge to completely avoid double outlines
    merge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.dilate(edges, merge_k, iterations=1)
    
    combined = cv2.bitwise_or(merged, ridges)
    thinned = _thin_edges(combined)
    
    cnts, _ = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in cnts:
        p = cv2.arcLength(cnt, False)
        if p < min_path_len:
            continue
        approx = cv2.approxPolyDP(cnt, 0.4, False)
        if len(approx) > 1:
            path = [tuple(pt[0]) for pt in approx]
            if len(path) > 1:
                smoothed = _smooth_path(path, window_size=3)
                paths.append(smoothed)
    return paths


def _generate_color_cross_contour_hatching(img_rgb, gray, saliency_norm, w, h):
    strokes = []
    spacing = max(6, min(w, h) // 90)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            sal = float(saliency_norm[y, x]) if saliency_norm is not None else 1.0
            if sal < 0.08:
                continue
                
            lum = int(gray[y, x])
            if lum >= 205:
                continue
                
            # Skin protection
            if sal > 0.45 and lum > 140:
                continue
                
            c_sampled = img_rgb[y, x].astype(np.float32)
            
            # Compute tangent direction
            dx = sobelx[y, x]
            dy = sobely[y, x]
            mag = np.sqrt(dx**2 + dy**2)
            
            if mag > 5.0:
                angle = np.arctan2(dy, dx) + np.pi / 2.0
            else:
                angle = np.radians(45.0)
                
            angle += np.radians(random.uniform(-10, 10))
            
            if lum < 80:
                lengths = [random.uniform(10, 18), random.uniform(8, 14)]
                angles = [angle, angle + np.pi / 2.0]
            else:
                lengths = [random.uniform(8, 14)]
                angles = [angle]
                
            for L, ang in zip(lengths, angles):
                cos_a = np.cos(ang)
                sin_a = np.sin(ang)
                x1 = x - cos_a * L / 2.0 + random.uniform(-0.5, 0.5)
                y1 = y - sin_a * L / 2.0 + random.uniform(-0.5, 0.5)
                x2 = x + cos_a * L / 2.0 + random.uniform(-0.5, 0.5)
                y2 = y + sin_a * L / 2.0 + random.uniform(-0.5, 0.5)
                
                if lum < 50:
                    stroke_factor = 0.05 + random.uniform(0, 0.04)  # Very dark colors
                    lw = 2
                elif lum < 100:
                    stroke_factor = 0.12 + random.uniform(0, 0.06)
                    lw = 2 if random.random() > 0.6 else 1
                else:
                    stroke_factor = 0.32 + random.uniform(0, 0.10)
                    lw = 1
                    
                stroke_rgb = tuple(max(3, int(c * stroke_factor)) for c in c_sampled)
                strokes.append(((x1, y1), (x2, y2), stroke_rgb, lw, y))
                
    strokes.sort(key=lambda s: s[4] + random.uniform(-30, 30))
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

    # Detect original highlights (catchlights) to preserve
    highlight_mask = (gray_raw > 225) & (saliency_norm > 0.35)
    highlight_mask = cv2.dilate(highlight_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # ── Canvas starts as pure white paper ─────────────────────────────────────
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    yield canvas.copy()

    # ── Step 1: Draw outline paths progressively ──────────────────────────────
    clo = max(15, 60 - threshold_c * 5)
    chi = max(45, 140 - threshold_c * 8)
    paths = _build_contour_paths(gray, saliency_norm, clo, chi, min_path_len=8)

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
                stroke_factor = 0.05 + random.uniform(0, 0.03)  # Rich dark outlines
                line_w = 3
            elif lum < 80:
                stroke_factor = 0.10 + random.uniform(0, 0.04)
                line_w = 2
            elif lum < 140:
                stroke_factor = 0.18 + random.uniform(0, 0.05)
                line_w = 2
            elif lum < 190:
                stroke_factor = 0.32 + random.uniform(0, 0.08)
                line_w = 2
            else:
                stroke_factor = 0.50 + random.uniform(0, 0.10)
                line_w = 1

        stroke_rgb = tuple(max(3, int(c * stroke_factor)) for c in c_sampled)

        # Only add jitter to background elements; keep face/subject precise
        pts = _jitter(path, jitter) if path_sal < 0.20 else path
        
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                draw_layer.line([pts[i], pts[i + 1]], fill=(*stroke_rgb, 255), width=line_w)

        if idx % eff_out == 0 or idx == n_out - 1:
            canvas_copy = canvas.copy()
            copy_np = np.array(canvas_copy)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGBA")

    # ── Step 2: Draw color cross-contour hatching strokes ─────────────────────
    hatching_strokes = _generate_color_cross_contour_hatching(img_np, gray, saliency_norm, w, h)
    n_hatch = len(hatching_strokes)
    eff_hatch = max(5, min(batch_size * 5, max(5, n_hatch // 100)))

    for idx, (pt1, pt2, stroke_rgb, lw, _) in enumerate(hatching_strokes):
        x1, y1 = pt1
        x2, y2 = pt2
        mx = (x1 + x2) / 2 + random.uniform(-0.4, 0.4)
        my = (y1 + y2) / 2 + random.uniform(-0.4, 0.4)
        
        draw_layer.line([pt1, (mx, my), pt2], fill=(*stroke_rgb, 255), width=lw)
        
        if idx % eff_hatch == 0 or idx == n_hatch - 1:
            canvas_copy = canvas.copy()
            copy_np = np.array(canvas_copy)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGBA")

    # ── Step 3: Fade in color wash base under outlines at the end ─────────────
    # Make sure highlight catchlights are completely preserved on lines
    canvas_np = np.array(canvas)
    canvas_np[highlight_mask > 0] = [255, 255, 255, 255]
    lines_np = canvas_np[:, :, :3]

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

    # Final Step: Paper texture
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
