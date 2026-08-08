"""
modes/anime.py — Anime / Manga Outline (5-Layer Progressive Drawing)
====================================================================
Approach: TRUE progressive anime/manga drawing flow with Gemini-Guided Layout
  - Layer 1: Structural Draft (Phác thảo cấu trúc):
    * Faint geometric guidelines (oval, axes, eyes) in light gray (235, 235, 235).
  - Layer 2: Main Silhouette & Outlines (Nét viền chính):
    * Silhouette outlines drawn in tapered black ink.
  - Layer 3: Face & Portrait Details (Chi tiết ngũ quan):
    * High-contrast detailed lineart for eyes, eyebrows, nose, mouth.
  - Layer 4: Cel Shaded Fill (Tô màu Cel-shading):
    * Cel colors fade in dynamically under outlines.
  - Layer 5: Eraser Phase (Xoá nét nháp):
    * Progressively fade out structural guidelines to 0 opacity.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw

from src.gpu_utils import (
    gpu_xdog,
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
    for _ in range(15):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _cel_quantise(img_np: np.ndarray, levels: int = 4) -> np.ndarray:
    step = 256 // levels
    q = (img_np.astype(np.float32) / step).astype(np.uint8) * step
    return np.clip(q + step // 2, 0, 255).astype(np.uint8)


def _draw_tapered_line(draw_layer, pts, color, base_width):
    N = len(pts)
    if N < 2:
        return
    for i in range(N - 1):
        t = (i + 0.5) / N
        w = max(1.0, base_width * np.sin(np.pi * t))
        draw_layer.line([pts[i], pts[i+1]], fill=color, width=int(round(w)))


def draw(
    pil_img: Image.Image,
    *,
    blur_size: int = 5,
    threshold_c: int = 10,
    jitter: float = 0.15,
    line_art_width: int = 2,
    batch_size: int = 10,
    **kw,
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
    saliency_norm = gpu_saliency(gray_raw)

    sigma1 = max(0.8, blur_size * 0.18)
    sigma2 = sigma1 * 1.6
    edge_mask = gpu_xdog(gray, sigma1=sigma1, sigma2=sigma2, tau=0.985, phi=16.0, epsilon=-0.05)

    # Clean double outlines and thin to centerline
    edges = 255 - edge_mask
    merged = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    thinned = _thin_edges(merged)
    edge_mask = 255 - thinned

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
    target_frames = max(250, min(1600, target_frames))

    # ── Layer 1: Structural Draft Guidelines ─────────────────────────────────
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

    # ── Drawing Canvas ────────────────────────────────────────────────────────
    drawing_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(drawing_canvas)

    paper_base = Image.new("RGB", (w, h), (255, 255, 255))
    
    # Yield initial draft
    frame_img = paper_base.copy()
    frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
    yield frame_img.convert("RGBA")

    # Find outlines
    cnts, _ = cv2.findContours(255 - edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in cnts:
        p = cv2.arcLength(cnt, False)
        if p < 12:
            continue
        approx = cv2.approxPolyDP(cnt, 0.4, False)
        if len(approx) > 1:
            path = [tuple(pt[0]) for pt in approx]
            if len(path) > 1:
                paths.append(_smooth_path(path, window_size=3))

    # Separate outlines
    main_outlines = []
    portrait_details = []
    
    for path in paths:
        path_sal = _path_saliency(path, saliency_norm, w, h)
        sx = max(0, min(w - 1, int(path[0][0])))
        sy = max(0, min(h - 1, int(path[0][1])))
        dist = np.sqrt((sx - fc[0])**2 + (sy - fc[1])**2)
        
        if path_sal > 0.35 or dist < R:
            portrait_details.append(path)
        else:
            main_outlines.append(path)

    frames_per_stage = target_frames // 5

    # ── Stage 2: Draw Main Outlines (Silhouette) ─────────────────────────────
    eff_out = max(1, len(main_outlines) // max(10, frames_per_stage))
    for idx, path in enumerate(main_outlines):
        _draw_tapered_line(draw_layer, path, (30, 30, 30, 255), max(1, line_art_width - 1))

        if idx % eff_out == 0 or idx == len(main_outlines) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            yield frame_img.convert("RGBA")

    # ── Stage 3: Draw Face & Portrait Details ────────────────────────────────
    eff_det = max(1, len(portrait_details) // max(10, frames_per_stage))
    for idx, path in enumerate(portrait_details):
        _draw_tapered_line(draw_layer, path, (10, 10, 10, 255), line_art_width)

        if idx % eff_det == 0 or idx == len(portrait_details) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            yield frame_img.convert("RGBA")

    # ── Stage 4: Fade in Cel Colors Under outlines ────────────────────────────
    # Render final clean outlines without landmarks
    lines_only = paper_base.copy()
    lines_only.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
    lines_np = np.array(lines_only)

    for step in range(1, frames_per_stage + 1):
        alpha = step / float(frames_per_stage)
        # Blend white background with cel color base
        base_color = cv2.addWeighted(np.full((h, w, 3), 255, dtype=np.uint8), 1.0 - alpha, cel, alpha, 0)
        # Multiply lines on top
        blended = np.clip((base_color.astype(np.float32) / 255.0) * (lines_np.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)
        
        frame_img = Image.fromarray(blended).convert("RGBA")
        # Overlay guidelines
        frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
        yield frame_img.copy()

    # ── Stage 5: Eraser Phase (progressive fade guidelines) ─────────────────
    eraser_steps = max(15, frames_per_stage)
    for step in range(eraser_steps):
        opacity_factor = 1.0 - (step / float(eraser_steps))
        
        # Base wash + lines
        base_color = cel.copy()
        blended = np.clip((base_color.astype(np.float32) / 255.0) * (lines_np.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)
        frame_img = Image.fromarray(blended).convert("RGBA")
        
        # Overlay faded draft
        draft_np = np.array(draft_canvas)
        draft_np[:, :, 3] = (draft_np[:, :, 3].astype(np.float32) * opacity_factor).astype(np.uint8)
        faded_draft = Image.fromarray(draft_np)
        
        frame_img.paste(faded_draft, (0, 0), mask=faded_draft.split()[3])
        yield frame_img.copy()
        faded_draft.close()

    # Final frame (clean, guidelines completely erased)
    blended = np.clip((cel.astype(np.float32) / 255.0) * (lines_np.astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)
    yield Image.fromarray(blended).convert("RGBA")
    
    paper_base.close()
    draft_canvas.close()
    drawing_canvas.close()
