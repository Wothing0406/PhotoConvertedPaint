"""
modes/pencil.py — Colored Pencil Sketch
========================================
Approach: MASTER progressive colored pencil technique
  1. Start with a soft, light color wash base (warm tinted paper, 40% opacity) from Frame 1.
  2. Layer 1: Structural Draft (Phác thảo cấu trúc):
     - Draw faint, light-gray geometric guidelines based on Gemini landmarks (face oval, tilt axes, eyes).
     - Underdrawing layout in faint gray.
  3. Layer 2: Detailed Linework (Nét vẽ chi tiết):
     - Precise color-preserved outlines drawn with tapered strokes.
     - Focal point detail: lines near the face landmarks are sharp, dark, and highly detailed.
     - Simplification (buông lỏng hậu cảnh): Farther lines simplified and lightened.
  4. Layer 3: Sweeping Contour Hatching (Đánh bóng & Đan nét):
     - Tangent-aligned long sweeping colored strokes.
     - Shading density scaled by the hatching slider.
  5. Eraser Phase (Xoá nét dư thừa):
     - In the final 20 frames, progressively fade out the light-gray structural guidelines (reduce opacity to 0).
  6. Apply linen paper grain texture.
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


def _dog_edges(gray, sigma=1.2, k=1.6, tau=0.98, thresh=8):
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1.astype(np.float32) - tau * g2.astype(np.float32)
    
    edges = np.zeros_like(gray)
    edges[dog < -thresh] = 255
    return edges


def _build_contour_paths(gray, saliency_norm, clo, chi, min_path_len=15):
    h, w = gray.shape

    # 1. Hessian Ridge
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.8)
    Ixx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
    val = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2))
    
    ridges = np.zeros_like(gray, dtype=np.uint8)
    ridges[val > 5.5] = 255

    # 2. DoG Edges
    edges = _dog_edges(gray, sigma=1.2, k=1.6, tau=0.97, thresh=6)
    
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
                mx, my = path[len(path)//2]
                rx, ry = max(0, min(w-1, int(mx))), max(0, min(h-1, int(my)))
                if gray[ry, rx] < 40 and saliency_norm[ry, rx] < 0.25:
                    continue
                    
                smoothed = _smooth_path(path, window_size=3)
                paths.append(smoothed)
    return paths


def _draw_tapered_line(draw_layer, pts, color, base_width):
    N = len(pts)
    if N < 2:
        return
    for i in range(N - 1):
        t = (i + 0.5) / N
        w = max(1.0, base_width * np.sin(np.pi * t))
        draw_layer.line([pts[i], pts[i+1]], fill=color, width=int(round(w)))


def _generate_color_cross_contour_hatching(img_rgb, gray, saliency_norm, w, h, hatching_intensity, fc, R):
    strokes = []
    spacing = max(12, min(w, h) // 70)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            sal = float(saliency_norm[y, x]) if saliency_norm is not None else 1.0
            
            dist = np.sqrt((x - fc[0])**2 + (y - fc[1])**2)
            sal_thresh = 0.22 if dist < R else 0.40
            if sal < sal_thresh:
                continue
                
            lum = int(gray[y, x])
            if lum >= 205:
                continue
                
            if lum >= 60 and hatching_intensity <= 0.05:
                continue
                
            if lum >= 60 and random.random() > hatching_intensity:
                continue
                
            if sal > 0.45 and lum > 140:
                continue
                
            c_sampled = img_rgb[y, x].astype(np.float32)
            
            dx = sobelx[y, x]
            dy = sobely[y, x]
            mag = np.sqrt(dx**2 + dy**2)
            
            if mag > 5.0:
                angle = np.arctan2(dy, dx) + np.pi / 2.0
            else:
                angle = np.radians(45.0)
                
            angle += np.radians(random.uniform(-8, 8))
            
            if lum < 60:
                L = random.uniform(30, 45)
            else:
                L = random.uniform(22, 32)
                
            if dist >= R:
                L *= 0.7
                
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            x1 = x - cos_a * L / 2.0 + random.uniform(-0.5, 0.5)
            y1 = y - sin_a * L / 2.0 + random.uniform(-0.5, 0.5)
            x2 = x + cos_a * L / 2.0 + random.uniform(-0.5, 0.5)
            y2 = y + sin_a * L / 2.0 + random.uniform(-0.5, 0.5)
            
            if lum < 50:
                stroke_factor = 0.05 + random.uniform(0, 0.03)
                lw = 2
            elif lum < 100:
                stroke_factor = 0.12 + random.uniform(0, 0.06)
                lw = 2 if random.random() > 0.7 else 1
            else:
                stroke_factor = 0.32 + random.uniform(0, 0.10)
                lw = 1
            
            factor = 0.35 + 0.62 * stroke_factor
            if dist >= R:
                factor = min(0.95, factor + 0.30) # lighter background shading
                lw = 1
                
            stroke_rgb = tuple(max(5, int(c * factor)) for c in c_sampled)
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
    hatching: float = 0.0,
    batch_size: int = 10,
    **kw,
):
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))
    gray_raw = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    gray = cv2.bilateralFilter(gray_raw, 7, 30, 30)
    saliency_norm = gpu_saliency(gray_raw)

    highlight_mask = (gray_raw > 225) & (saliency_norm > 0.35)
    highlight_mask = cv2.dilate(highlight_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # ── Parse Gemini face landmarks ───────────────────────────────────────────
    face_cx = float(kw.get("face_center_x", 0.5))
    face_cy = float(kw.get("face_center_y", 0.4))
    face_w = float(kw.get("face_width", 0.3))
    face_h = float(kw.get("face_height", 0.45))
    tilt = float(kw.get("head_tilt_angle", 0.0))

    fc = (face_cx * w, face_cy * h)
    R = max(face_w * w, face_h * h) * 1.1

    bs = max(5, min(50, batch_size))
    target_frames = int(1600 - (bs - 5) * (1300 / 45))
    target_frames = max(200, min(1600, target_frames))

    # ── Create transparent Layer for Structural Draft Guidelines ──────────────
    draft_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draft_draw = ImageDraw.Draw(draft_canvas)
    guide_color = (225, 225, 225, 255) # faint gray
    
    # 1. Head Oval
    draft_draw.ellipse(
        [fc[0] - face_w * w / 2, fc[1] - face_h * h / 2, fc[0] + face_w * w / 2, fc[1] + face_h * h / 2],
        outline=guide_color, width=1
    )
    # 2. Main Vertical Axis
    rad = np.radians(tilt)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    draft_draw.line(
        [fc[0] - sin_t * face_h * h / 2, fc[1] - cos_t * face_h * h / 2,
         fc[0] + sin_t * face_h * h / 2, fc[1] + cos_t * face_h * h / 2],
        fill=guide_color, width=1
    )
    # 3. Eye line
    draft_draw.line(
        [fc[0] - cos_t * face_w * w / 2, fc[1] + sin_t * face_w * w / 2,
         fc[0] + cos_t * face_w * w / 2, fc[1] - sin_t * face_w * w / 2],
        fill=guide_color, width=1
    )

    # ── Drawing Layer ─────────────────────────────────────────────────────────
    drawing_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(drawing_canvas)

    # ── Color Wash Base from Frame 1 ──────────────────────────────────────────
    blur_r = max(20, min(w, h) // 5)
    washed = np.array(
        pil_img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_r))
    ).astype(np.float32)
    wash_blended = np.clip(255.0 * 0.60 + washed * 0.40, 0, 255).astype(np.uint8)
    
    paper_base = Image.fromarray(wash_blended).convert("RGBA")
    
    # Initial Frame: Show paper wash + draft guidelines
    frame_img = paper_base.copy()
    frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
    yield frame_img.convert("RGBA")

    # ── Step 1: Draw outlines ─────────────────────────────────────────────────
    clo = max(15, 60 - threshold_c * 5)
    chi = max(45, 140 - threshold_c * 8)
    paths = _build_contour_paths(gray, saliency_norm, clo, chi, min_path_len=15)
    paths.sort(key=lambda p: _path_saliency(p, saliency_norm, w, h), reverse=True)

    n_out = len(paths)
    target_out_frames = max(30, target_frames // 4)
    eff_out = max(1, n_out // target_out_frames)

    for idx, path in enumerate(paths):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        c_sampled = img_np[sy, sx].astype(np.float32)
        lum = float(gray[sy, sx])
        path_sal = _path_saliency(path, saliency_norm, w, h)
        
        dist = np.sqrt((sx - fc[0])**2 + (sy - fc[1])**2)

        if path_sal < 0.10:
            if len(path) < 15:
                continue
            stroke_factor = 0.80 + random.uniform(0, 0.15)
            line_w = 1
        else:
            if lum < 40:
                stroke_factor = 0.05 + random.uniform(0, 0.03)
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

        if dist >= R:
            stroke_factor = min(0.95, stroke_factor + 0.35) # soften background borders
            line_w = 1

        factor = 0.35 + 0.62 * stroke_factor
        stroke_rgb = tuple(max(5, int(c * factor)) for c in c_sampled)
        
        _draw_tapered_line(draw_layer, path, (*stroke_rgb, 255), line_w)

        if idx % eff_out == 0 or idx == n_out - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGBA")

    # ── Step 2: Draw color hatching ───────────────────────────────────────────
    hatching_strokes = _generate_color_cross_contour_hatching(img_np, gray, saliency_norm, w, h, hatching, fc, R)
    n_hatch = len(hatching_strokes)
    target_hatch_frames = max(70, target_frames * 3 // 4)
    eff_hatch = max(1, n_hatch // target_hatch_frames)

    for idx, (pt1, pt2, stroke_rgb, lw, _) in enumerate(hatching_strokes):
        x1, y1 = pt1
        x2, y2 = pt2
        mx = (x1 + x2) / 2 + random.uniform(-0.4, 0.4)
        my = (y1 + y2) / 2 + random.uniform(-0.4, 0.4)
        
        draw_layer.line([pt1, (mx, my), pt2], fill=(*stroke_rgb, 255), width=lw)
        
        if idx % eff_hatch == 0 or idx == n_hatch - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGBA")

    # ── Step 3: Eraser Phase (Fade out structural draft layer) ────────────────
    eraser_steps = 15
    for step in range(eraser_steps):
        opacity_factor = 1.0 - (step / float(eraser_steps))
        
        frame_img = paper_base.copy()
        
        draft_np = np.array(draft_canvas)
        draft_np[:, :, 3] = (draft_np[:, :, 3].astype(np.float32) * opacity_factor).astype(np.uint8)
        faded_draft = Image.fromarray(draft_np)
        
        frame_img.paste(faded_draft, (0, 0), mask=faded_draft.split()[3])
        frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
        
        copy_np = np.array(frame_img)
        copy_np[highlight_mask > 0] = [255, 255, 255, 255]
        yield Image.fromarray(copy_np).convert("RGBA")

    # Final Step: Blend paper texture on top
    canvas_np = np.array(paper_base)
    # Paste clean drawing
    drawing_np = np.array(drawing_canvas)
    # manual blend
    mask = drawing_np[:, :, 3] > 0
    canvas_np[mask] = drawing_np[mask]
    canvas_np[highlight_mask > 0] = [255, 255, 255, 255]
    final_canvas = Image.fromarray(canvas_np)
    
    try:
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
        yield final_canvas.copy()

    paper_base.close()
    draft_canvas.close()
    drawing_canvas.close()
