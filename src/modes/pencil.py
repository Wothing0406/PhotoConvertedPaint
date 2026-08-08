"""
modes/pencil.py — Colored Pencil Sketch (5-Layer Progressive Drawing)
======================================================================
Approach: MASTER progressive colored pencil technique in 5 stages
  - Preprocessing: Focal-Saliency Edge Guidance:
    * Apply aggressive bilateral filter/blur outside the face area to completely
      eliminate noisy background textures (like flowers, garments) before edge detection.
  - Path Extraction: 1D Gaussian Path Smoothing:
    * Keep raw coordinates and smooth them with a 1D Gaussian kernel to ensure
      silky, organic curves (eliminating polygonal distortion).
  - Line Weight & Tone Tapering:
    * Vary both stroke width AND pencil color opacity along the path using a sine pressure curve.
  - Layer 1: Structural Draft & Colored Smudge Wash (Phác thảo & Di chì màu):
    * Faint geometric guidelines (oval, axes, eyes) based on Gemini landmarks.
    * Warm tinted color wash underpainting (65% opacity) to establish soft forms.
  - Layer 2: Silhouette & Main Outlines (Nét viền chính):
    * Silhouette and background contours drawn in color.
  - Layer 3: Face & Portrait Details (Chi tiết ngũ quan):
    * Dark, sharp, hue-preserved outlines for eyes, eyebrows, nose, mouth.
  - Layer 4: Core Shading & Dark Hatching (Đánh bóng tối):
    * Tangent-aligned colored sweeping strokes in deep shadows.
  - Layer 5: Midtone Shading & Eraser Clean-up (Đánh bóng sáng & Xoá nét nháp):
    * Soft, light color shading on the face/skin to give volume.
    * Progressively fade out the Layer 1 guidelines to 0 opacity.
    * Apply linen paper grain texture.
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


def _gaussian_smooth_path(path, sigma=1.5):
    N = len(path)
    if N < 3:
        return path
        
    radius = int(round(3.0 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    
    xs = np.array([pt[0] for pt in path])
    ys = np.array([pt[1] for pt in path])
    
    xs_padded = np.pad(xs, radius, mode='edge')
    ys_padded = np.pad(ys, radius, mode='edge')
    
    xs_smooth = np.convolve(xs_padded, kernel, mode='valid')
    ys_smooth = np.convolve(ys_padded, kernel, mode='valid')
    
    return [ (float(xs_smooth[i]), float(ys_smooth[i])) for i in range(N) ]


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

    # 1. Focal-Saliency Preprocessing:
    # Build a hybrid source image for edge detection where background/flowers are heavily smoothed
    # to prevent noisy textured outlines, while keeping the face crisp.
    x_indices = np.arange(w)
    y_indices = np.arange(h)
    xs, ys = np.meshgrid(x_indices, y_indices)
    dists = np.sqrt((xs - fc[0])**2 + (ys - fc[1])**2)
    
    # Focal mask
    focal_mask = np.clip(1.0 - (dists - R) / R, 0.0, 1.0)
    focal_mask = cv2.GaussianBlur(focal_mask, (21, 21), 0)
    
    # Create heavily smoothed background image
    bg_smooth = cv2.bilateralFilter(gray, d=17, sigmaColor=120, sigmaSpace=120)
    bg_smooth = cv2.GaussianBlur(bg_smooth, (9, 9), 0)
    
    # Combine sharp face and smooth background
    hybrid_gray = (gray.astype(np.float32) * focal_mask + bg_smooth.astype(np.float32) * (1.0 - focal_mask)).astype(np.uint8)

    # 2. Extract edges from hybrid
    blurred = cv2.GaussianBlur(hybrid_gray, (5, 5), 0.8)
    Ixx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
    val = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2))
    
    ridges = np.zeros_like(gray, dtype=np.uint8)
    ridges[val > 5.5] = 255

    edges = _dog_edges(hybrid_gray, sigma=1.2, k=1.6, tau=0.97, thresh=6)
    
    merge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.dilate(edges, merge_k, iterations=1)
    
    combined = cv2.bitwise_or(merged, ridges)
    thinned = _thin_edges(combined)
    
    cnts, _ = cv2.findContours(thinned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in cnts:
        p = cv2.arcLength(cnt, False)
        approx = cv2.approxPolyDP(cnt, 0.25, False)
        if len(approx) > 1:
            path = [tuple(pt[0]) for pt in approx]
            mx, my = path[len(path)//2]
            dist = np.sqrt((mx - fc[0])**2 + (my - fc[1])**2)
            
            min_len = 6 if dist < R else 18
            if p < min_len:
                continue
                
            rx, ry = max(0, min(w-1, int(mx))), max(0, min(h-1, int(my)))
            if gray[ry, rx] < 40 and saliency_norm[ry, rx] < 0.25:
                continue
                
            smoothed = _gaussian_smooth_path(path, sigma=2.0)
            paths.append(smoothed)
    return paths


def _draw_tapered_line(draw_layer, pts, base_rgb, base_width):
    """Draws a smooth colored line with variable stroke width AND color opacity (pressure tapering)."""
    N = len(pts)
    if N < 2:
        return
    for i in range(N - 1):
        t = (i + 0.5) / N
        pressure = np.sin(np.pi * t)
        
        w = max(1.0, base_width * pressure)
        
        # Color pressure blend (fades into paper color 255 at the ends)
        r = int(round(255 - (255 - base_rgb[0]) * pressure))
        g = int(round(255 - (255 - base_rgb[1]) * pressure))
        b = int(round(255 - (255 - base_rgb[2]) * pressure))
        
        draw_layer.line([pts[i], pts[i+1]], fill=(r, g, b, 255), width=int(round(w)))


def _generate_color_cross_contour_hatching(img_rgb, gray, saliency_norm, w, h, hatching_intensity, fc, R):
    strokes = []
    spacing = max(12, min(w, h) // 80)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            lum = int(gray[y, x])
            if lum >= 215:
                continue
                
            dist = np.sqrt((x - fc[0])**2 + (y - fc[1])**2)
            sal = float(saliency_norm[y, x]) if saliency_norm is not None else 1.0
            
            # If it's a dark shadow region (e.g. hair, dark clothes, shadow folds),
            # we ALWAYS hatch it to build the dark values of the sketch, bypassing saliency gating.
            if lum >= 85:
                # For midtones/lights, we use saliency gating to keep background clean
                sal_thresh = 0.20 if dist < R else 0.45
                if sal < sal_thresh:
                    continue
                    
                # Midtone hatching slider check
                if hatching_intensity <= 0.05:
                    continue
                if random.random() > hatching_intensity:
                    continue
            
            c_sampled = img_rgb[y, x].astype(np.float32)
            is_face = dist < R

            if is_face and lum > 140:
                if random.random() > 0.25:
                    continue
                stroke_factor = 0.65 + random.uniform(0, 0.10)
                lw = 1
            else:
                if lum < 50:
                    stroke_factor = 0.05 + random.uniform(0, 0.03)
                    lw = 2
                elif lum < 100:
                    stroke_factor = 0.12 + random.uniform(0, 0.06)
                    lw = 2 if random.random() > 0.7 else 1
                else:
                    stroke_factor = 0.32 + random.uniform(0, 0.10)
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
                L *= 0.6
                stroke_factor = min(0.95, stroke_factor + 0.35)
                lw = 1
                
            factor = 0.35 + 0.62 * stroke_factor
            stroke_rgb = tuple(max(5, int(c * factor)) for c in c_sampled)
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            pt1 = (x - cos_a * L / 2, y - sin_a * L / 2)
            pt2 = (x + cos_a * L / 2, y + sin_a * L / 2)
            
            strokes.append((pt1, pt2, stroke_rgb, lw, y, lum))
                
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

    # ── Layer 1: Structural Draft Guidelines & Colored Smudge Wash ───────────
    draft_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draft_draw = ImageDraw.Draw(draft_canvas)
    guide_color = (225, 225, 225, 255)
    
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

    # ── Color Wash Base from Frame 1 ──────────────────────────────────────────
    blur_r = max(20, min(w, h) // 5)
    washed = np.array(
        pil_img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_r))
    ).astype(np.float32)
    wash_blended = np.clip(255.0 * 0.40 + washed * 0.60, 0, 255).astype(np.uint8)
    
    paper_base = Image.fromarray(wash_blended).convert("RGBA")
    
    # ── Drawing Canvas ────────────────────────────────────────────────────────
    drawing_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(drawing_canvas)
    
    # Yield initial wash + draft
    frame_img = paper_base.copy()
    frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
    yield frame_img.convert("RGBA")

    # Get outline paths
    all_paths = _build_contour_paths(gray, saliency_norm, None, None, fc, R)
    
    main_outlines = []
    portrait_details = []
    
    for path in all_paths:
        path_sal = _path_saliency(path, saliency_norm, w, h)
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        dist = np.sqrt((sx - fc[0])**2 + (sy - fc[1])**2)
        
        if path_sal > 0.35 or dist < R:
            portrait_details.append(path)
        else:
            main_outlines.append(path)

    # Hatching splits
    all_hatching = _generate_color_cross_contour_hatching(img_np, gray, saliency_norm, w, h, hatching, fc, R)
    dark_hatching = [h for h in all_hatching if h[5] < 85]
    midtone_hatching = [h for h in all_hatching if h[5] >= 85]

    frames_per_stage = target_frames // 5

    # ── Stage 2: Draw Main Outlines (Silhouette) ─────────────────────────────
    eff_out = max(1, len(main_outlines) // max(10, frames_per_stage))
    for idx, path in enumerate(main_outlines):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        c_sampled = img_np[sy, sx].astype(np.float32)
        dist = np.sqrt((sx - fc[0])**2 + (sy - fc[1])**2)

        stroke_factor = 0.50 + random.uniform(0, 0.10)
        line_w = 1
        if dist >= R:
            stroke_factor = min(0.95, stroke_factor + 0.35)

        factor = 0.35 + 0.62 * stroke_factor
        stroke_rgb = tuple(max(5, int(c * factor)) for c in c_sampled)

        _draw_tapered_line(draw_layer, path, stroke_rgb, line_w)

        if idx % eff_out == 0 or idx == len(main_outlines) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGBA")

    # ── Stage 3: Draw Face & Portrait Details ────────────────────────────────
    eff_det = max(1, len(portrait_details) // max(10, frames_per_stage))
    for idx, path in enumerate(portrait_details):
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        c_sampled = img_np[sy, sx].astype(np.float32)
        lum = float(gray[sy, sx])

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

        factor = 0.35 + 0.62 * stroke_factor
        stroke_rgb = tuple(max(5, int(c * factor)) for c in c_sampled)

        _draw_tapered_line(draw_layer, path, stroke_rgb, line_w)

        if idx % eff_det == 0 or idx == len(portrait_details) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGBA")

    # ── Stage 4: Draw Core Shading & Dark Hatching ───────────────────────────
    eff_dark = max(1, len(dark_hatching) // max(10, frames_per_stage))
    for idx, (pt1, pt2, stroke_rgb, lw, _, _) in enumerate(dark_hatching):
        mx = (pt1[0] + pt2[0]) / 2 + random.uniform(-0.4, 0.4)
        my = (pt1[1] + pt2[1]) / 2 + random.uniform(-0.4, 0.4)
        
        draw_layer.line([pt1, (mx, my), pt2], fill=(*stroke_rgb, 255), width=lw)
        
        if idx % eff_dark == 0 or idx == len(dark_hatching) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGBA")

    # ── Stage 5: Draw Midtone Shading & Fading Draft guidelines ──────────────
    eff_mid = max(1, len(midtone_hatching) // max(10, frames_per_stage))
    total_mid = len(midtone_hatching)
    
    for idx, (pt1, pt2, stroke_rgb, lw, _, _) in enumerate(midtone_hatching):
        mx = (pt1[0] + pt2[0]) / 2 + random.uniform(-0.4, 0.4)
        my = (pt1[1] + pt2[1]) / 2 + random.uniform(-0.4, 0.4)
        
        draw_layer.line([pt1, (mx, my), pt2], fill=(*stroke_rgb, 255), width=lw)
        
        if idx % eff_mid == 0 or idx == total_mid - 1:
            opacity_factor = 1.0 - (idx / float(max(1, total_mid - 1)))
            
            frame_img = paper_base.copy()
            draft_np = np.array(draft_canvas)
            draft_np[:, :, 3] = (draft_np[:, :, 3].astype(np.float32) * opacity_factor).astype(np.uint8)
            faded_draft = Image.fromarray(draft_np)
            
            frame_img.paste(faded_draft, (0, 0), mask=faded_draft.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            copy_np = np.array(frame_img)
            copy_np[highlight_mask > 0] = [255, 255, 255, 255]
            yield Image.fromarray(copy_np).convert("RGBA")
            faded_draft.close()

    # Final Step: Blend paper texture
    canvas_np = np.array(paper_base)
    drawing_np = np.array(drawing_canvas)
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
