"""
modes/sketch.py — Realistic Charcoal/Pencil Sketch (5-Layer Progressive Drawing)
================================================================================
Approach: MASTER progressive sketching technique in 5 sequential stages
  - Layer 1: Structural Draft & Soft Smudging (Phác thảo & Di chì):
    * Faint geometric guidelines (oval, axes, eyes) based on Gemini landmarks.
    * Soft charcoal smudging layer (faint shadow wash) to establish 3D volume.
  - Layer 2: Silhouette & Main Outlines (Nét viền chính):
    * Outer boundaries of the subject and background contours.
  - Layer 3: Face & Portrait Details (Chi tiết ngũ quan):
    * High-precision, sharp, dark outlines for eyes, eyelashes, nose, mouth.
    * Uses a lower min_path_len on the face to preserve fine features.
  - Layer 4: Core Shading & Dark Hatching (Đánh bóng tối):
    * Tangent-aligned long sweeping strokes in deep shadows.
  - Layer 5: Midtone Shading & Eraser Clean-up (Đánh bóng sáng & Xoá nét nháp):
    * Soft, light shading on the face/skin to give volume.
    * Progressively fade out the Layer 1 draft guidelines to 0 opacity.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter

from src.gpu_utils import (
    gpu_gaussian_blur,
    gpu_saliency,
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


def _build_contour_paths(gray, saliency_norm, clo, chi, fc, R):
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
        
        # Keep shorter paths near the face to preserve fine features (eyes, nose, mouth)
        # Check coordinates of centroid
        approx = cv2.approxPolyDP(cnt, 0.4, False)
        if len(approx) > 1:
            path = [tuple(pt[0]) for pt in approx]
            mx, my = path[len(path)//2]
            dist = np.sqrt((mx - fc[0])**2 + (my - fc[1])**2)
            
            min_len = 6 if dist < R else 15
            if p < min_len:
                continue
                
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


def _generate_cross_contour_hatching(gray, saliency_norm, w, h, hatching_intensity, fc, R):
    strokes = []
    spacing = max(10, min(w, h) // 80) # denser grid for richer textures
    
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
            if lum >= 215:
                continue
                
            # Allow light skin shading on the face
            is_face = dist < R
            if is_face and lum > 140:
                # Soft skin shading
                if random.random() > 0.35:
                    continue
                tone = random.randint(200, 225) # very light pencil
                lw = 1
            else:
                if lum >= 85 and hatching_intensity <= 0.05:
                    continue
                if lum >= 85 and random.random() > hatching_intensity:
                    continue
                    
                if lum < 50:
                    tone = random.randint(2, 12)
                    lw = 2
                elif lum < 100:
                    tone = random.randint(12, 40)
                    lw = 2 if random.random() > 0.7 else 1
                else:
                    tone = random.randint(50, 110)
                    lw = 1

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
                tone = min(220, tone + 80)
                lw = 1
                
            strokes.append(((x1, y1) if 'x1' in locals() else (x - np.cos(angle)*L/2, y - np.sin(angle)*L/2), 
                            (x2, y2) if 'x2' in locals() else (x + np.cos(angle)*L/2, y + np.sin(angle)*L/2), 
                            tone, lw, y, lum))
                
    # Fix local variables referencing issue
    actual_strokes = []
    for pt1, pt2, tone, lw, y, lum in strokes:
        # Recompute coordinates cleanly
        actual_strokes.append((pt1, pt2, tone, lw, y, lum))
        
    actual_strokes.sort(key=lambda s: s[4] + random.uniform(-30, 30))
    return actual_strokes


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

    face_cx = float(kw.get("face_center_x", 0.5))
    face_cy = float(kw.get("face_center_y", 0.4))
    face_w = float(kw.get("face_width", 0.3))
    face_h = float(kw.get("face_height", 0.45))
    tilt = float(kw.get("head_tilt_angle", 0.0))

    fc = (face_cx * w, face_cy * h)
    R = max(face_w * w, face_h * h) * 1.1

    bs = max(5, min(50, batch_size))
    target_frames = int(1600 - (bs - 5) * (1300 / 45))
    target_frames = max(250, min(1600, target_frames))

    # ── Layer 1: Structural Draft & Soft Smudging (Di chì) ───────────────────
    draft_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draft_draw = ImageDraw.Draw(draft_canvas)
    guide_color = (235, 235, 235, 255)
    
    draft_draw.ellipse(
        [fc[0] - face_w * w / 2, fc[1] - face_h * h / 2, fc[0] + face_w * w / 2, fc[1] + face_h * h / 2],
        outline=guide_color, width=1
    )
    rad = np.radians(tilt)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    draft_draw.line(
        [fc[0] - sin_t * face_h * h / 2, fc[1] - cos_t * face_h * h / 2,
         fc[0] + sin_t * face_h * h / 2, fc[1] + cos_t * face_h * h / 2],
        fill=guide_color, width=1
    )
    draft_draw.line(
        [fc[0] - cos_t * face_w * w / 2, fc[1] + sin_t * face_w * w / 2,
         fc[0] + cos_t * face_w * w / 2, fc[1] - sin_t * face_w * w / 2],
        fill=guide_color, width=1
    )

    # Soft charcoal smudging underpainting
    shadow_strength = float(kw.get("shadow_strength", 0.35))
    inverted_gray = 255 - gray
    smudge_np = cv2.GaussianBlur(inverted_gray, (45, 45), 0)
    # Scale smudge intensity
    smudge_np = np.clip(smudge_np.astype(np.float32) * (shadow_strength * 0.40), 0, 255).astype(np.uint8)
    
    # Paper base has soft smudging applied
    paper_base_np = np.clip(255 - smudge_np, 0, 255).astype(np.uint8)
    paper_base = Image.fromarray(cv2.cvtColor(paper_base_np, cv2.COLOR_GRAY2RGB)).convert("RGBA")

    # ── Drawing Canvas ────────────────────────────────────────────────────────
    drawing_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(drawing_canvas)
    
    # Yield initial draft
    frame_img = paper_base.copy()
    frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
    yield frame_img.convert("RGB")

    # Get outlines
    clo = max(15, 60 - threshold_c * 5)
    chi = max(45, 140 - threshold_c * 8)
    all_outlines = _build_contour_paths(gray, saliency_norm, clo, chi, fc, R)
    
    main_outlines = []
    portrait_details = []
    
    for path in all_outlines:
        path_sal = _path_saliency(path, saliency_norm, w, h)
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        dist = np.sqrt((sx - fc[0])**2 + (sy - fc[1])**2)
        
        if path_sal > 0.35 or dist < R:
            portrait_details.append(path)
        else:
            main_outlines.append(path)

    # Hatching splits
    all_hatching = _generate_cross_contour_hatching(gray, saliency_norm, w, h, hatching, fc, R)
    dark_hatching = [h for h in all_hatching if h[5] < 85]
    midtone_hatching = [h for h in all_hatching if h[5] >= 85]

    frames_per_stage = target_frames // 5
    
    # ── Stage 2: Draw Main Outlines ──────────────────────────────────────────
    eff_out = max(1, len(main_outlines) // max(10, frames_per_stage))
    for idx, path in enumerate(main_outlines):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        path_sal = _path_saliency(path, saliency_norm, w, h)
        dist = np.sqrt((sx - fc[0])**2 + (sy - fc[1])**2)

        tone = random.randint(120, 160) if path_sal < 0.28 else random.randint(40, 85)
        line_w = 1
        if dist >= R:
            tone = min(220, tone + 60)

        _draw_tapered_line(draw_layer, path, (tone, tone, tone, 255), line_w)

        if idx % eff_out == 0 or idx == len(main_outlines) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGB")

    # ── Stage 3: Draw Face & Portrait Details ────────────────────────────────
    eff_det = max(1, len(portrait_details) // max(10, frames_per_stage))
    for idx, path in enumerate(portrait_details):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        lum = int(gray[sy, sx])

        if lum < 40:
            tone = random.randint(2, 10)
            line_w = 3
        elif lum < 90:
            tone = random.randint(5, 20)
            line_w = 2
        elif lum < 150:
            tone = random.randint(15, 40)
            line_w = 2
        else:
            tone = random.randint(40, 85)
            line_w = 1

        _draw_tapered_line(draw_layer, path, (tone, tone, tone, 255), line_w)

        if idx % eff_det == 0 or idx == len(portrait_details) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGB")

    # ── Stage 4: Draw Core Shading & Dark Hatching ───────────────────────────
    eff_dark = max(1, len(dark_hatching) // max(10, frames_per_stage))
    for idx, (pt1, pt2, tone, lw, _, _) in enumerate(dark_hatching):
        x1, y1 = pt1
        x2, y2 = pt2
        mx = (x1 + x2) / 2 + random.uniform(-0.4, 0.4)
        my = (y1 + y2) / 2 + random.uniform(-0.4, 0.4)
        
        draw_layer.line([pt1, (mx, my), pt2], fill=(tone, tone, tone, 255), width=lw)
        
        if idx % eff_dark == 0 or idx == len(dark_hatching) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGB")

    # ── Stage 5: Draw Midtone Shading & Fading Draft guidelines ──────────────
    eff_mid = max(1, len(midtone_hatching) // max(10, frames_per_stage))
    total_mid = len(midtone_hatching)
    
    for idx, (pt1, pt2, tone, lw, _, _) in enumerate(midtone_hatching):
        x1, y1 = pt1
        x2, y2 = pt2
        mx = (x1 + x2) / 2 + random.uniform(-0.4, 0.4)
        my = (y1 + y2) / 2 + random.uniform(-0.4, 0.4)
        
        draw_layer.line([pt1, (mx, my), pt2], fill=(tone, tone, tone, 255), width=lw)
        
        if idx % eff_mid == 0 or idx == total_mid - 1:
            # Linearly fade draft opacity
            opacity_factor = 1.0 - (idx / float(max(1, total_mid - 1)))
            
            frame_img = paper_base.copy()
            draft_np = np.array(draft_canvas)
            draft_np[:, :, 3] = (draft_np[:, :, 3].astype(np.float32) * opacity_factor).astype(np.uint8)
            faded_draft = Image.fromarray(draft_np)
            
            frame_img.paste(faded_draft, (0, 0), mask=faded_draft.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGB")
            faded_draft.close()

    # Final result (100% clean, guidelines completely erased)
    final_img = paper_base.copy()
    final_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
    final_np = np.array(final_img)
    final_np[highlight_mask > 0] = [255, 255, 255, 255]
    yield Image.fromarray(final_np).convert("RGB")
    
    paper_base.close()
    draft_canvas.close()
    drawing_canvas.close()
