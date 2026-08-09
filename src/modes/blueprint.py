"""
modes/blueprint.py — Paint-by-Numbers Blueprint (5-Layer Progressive Drawing)
=============================================================================
Approach: TRUE paint-by-numbers painting flow in 5 sequential stages
  - Layer 1: Structural Draft (Phác thảo cấu trúc):
    * Faint light-blue guidelines (oval, axes, eyes) based on Gemini landmarks.
  - Layer 2: Region Outlines (Đường biên phân vùng):
    * Dark gray contour boundaries of all segments.
  - Layer 3: Numeric Labels (Đánh số màu):
    * Region color numbers drawn at the centroid of each segment.
  - Layer 4: Progressive Color Fill (Tô màu phân vùng):
    * Region color fills drawn progressively (largest to smallest).
  - Layer 5: Eraser Phase (Xoá nét nháp):
    * Fading out the light-blue blueprint draft guidelines.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont


def _kmeans_seg(img_rgb: np.ndarray, k: int = 12):
    h, w = img_rgb.shape[:2]
    smooth = cv2.bilateralFilter(img_rgb, d=15, sigmaColor=120, sigmaSpace=120)
    smooth = cv2.medianBlur(smooth, 11)
    
    pixels = smooth.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    
    centers = np.uint8(centers)
    seg = labels.reshape(h, w).astype(np.uint8)
    
    seg = cv2.medianBlur(seg, 13)
    seg = cv2.medianBlur(seg, 9)
    return seg, centers


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


def _region_boundaries(seg: np.ndarray) -> np.ndarray:
    h, w = seg.shape
    boundary = np.zeros((h, w), dtype=np.uint8)
    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        shifted = np.roll(seg, (dy, dx), axis=(0, 1))
        boundary = np.where(seg != shifted, 255, boundary).astype(np.uint8)
    return boundary


def draw(
    pil_img: Image.Image,
    *,
    jitter: float = 0.05,
    batch_size: int = 10,
    **kw,
):
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # Region segmentation
    k = 12
    seg, centers = _kmeans_seg(img_np, k=k)

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

    # ── Layer 1: Blue Draft guidelines (simulating blueprints) ────────────────
    draft_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draft_draw = ImageDraw.Draw(draft_canvas)
    guide_color = (200, 220, 255, 255) # faint blue
    
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

    paper_base = Image.new("RGB", (w, h), (252, 250, 244))
    
    # Yield initial paper with draft guidelines
    frame_img = paper_base.copy()
    frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
    yield frame_img.copy()

    # Extract boundaries
    boundary_mask = _region_boundaries(seg)
    thinned_b = _thin_edges(boundary_mask)
    cnts_b, _ = cv2.findContours(thinned_b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_b = sorted(cnts_b, key=lambda c: cv2.arcLength(c, False), reverse=True)

    frames_per_stage = target_frames // 5

    # ── Stage 2: Draw Outlines ───────────────────────────────────────────────
    eff_b = max(1, len(cnts_b) // max(10, frames_per_stage))
    for idx, cnt in enumerate(cnts_b):
        p = cv2.arcLength(cnt, False)
        if p < 12:
            continue
        approx = cv2.approxPolyDP(cnt, max(0.2, 0.003 * p), False)
        path = [tuple(pt[0]) for pt in approx]
        if len(path) > 1:
            result = []
            dx, dy = 0.0, 0.0
            for x, y in path:
                dx = 0.8 * dx + 0.2 * random.gauss(0, jitter)
                dy = 0.8 * dy + 0.2 * random.gauss(0, jitter)
                result.append((x + dx, y + dy))
            draw_layer.line(result, fill=(130, 130, 130, 255), width=1)
            
        if idx % eff_b == 0 or idx == len(cnts_b) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            yield frame_img.copy()

    # ── Stage 3: Draw Numbers ────────────────────────────────────────────────
    # Get centroids and colors for all unique segments
    regions = []
    for label_id in range(k):
        mask = (seg == label_id).astype(np.uint8)
        c_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cc in c_cnts:
            area = cv2.contourArea(cc)
            if area < 100:
                continue
            M = cv2.moments(cc)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                regions.append((area, cx, cy, label_id, cc))

    regions.sort(key=lambda r: r[0], reverse=True)
    eff_n = max(1, len(regions) // max(5, frames_per_stage))

    for idx, (area, cx, cy, label_id, cc) in enumerate(regions):
        num_str = str(label_id + 1)
        draw_layer.text((cx - 3, cy - 5), num_str, fill=(140, 140, 140, 255))
        
        if idx % eff_n == 0 or idx == len(regions) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            yield frame_img.copy()

    # Save the blank printable blueprint sheet (outlines + numbers on warm white paper)
    session_id = kw.get("session_id")
    if session_id:
        try:
            sheet_img = paper_base.copy()
            sheet_img.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            sheet_img.convert("RGB").save(f"output/blueprint_{session_id}.png")
            sheet_img.close()
            print(f"[Blueprint] Saved blank blueprint sheet: output/blueprint_{session_id}.png")
        except Exception as e:
            print(f"[Blueprint] Error saving blank sheet: {e}")

    # ── Stage 4: Progressive color filling ────────────────────────────────────
    filled_canvas = drawing_canvas.copy()
    fill_draw = ImageDraw.Draw(filled_canvas)

    for idx, (area, cx, cy, label_id, cc) in enumerate(regions):
        pts = [tuple(pt[0]) for pt in cc]
        color = tuple(centers[label_id])
        if len(pts) > 2:
            fill_draw.polygon(pts, fill=(*color, 255))

        if idx % eff_n == 0 or idx == len(regions) - 1:
            frame_img = paper_base.copy()
            frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
            frame_img.paste(filled_canvas, (0, 0), mask=filled_canvas.split()[3])
            # multiply outlines/numbers on top
            lines_only = paper_base.copy()
            lines_only.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])
            
            blended = np.clip((np.array(frame_img).astype(np.float32) / 255.0) * (np.array(lines_only).astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)
            yield Image.fromarray(blended)

    # ── Stage 5: Eraser Phase (fading blueprint draft guides) ────────────────
    lines_only = paper_base.copy()
    lines_only.paste(drawing_canvas, (0, 0), mask=drawing_canvas.split()[3])

    base_frame = paper_base.copy()
    base_frame.paste(filled_canvas, (0, 0), mask=filled_canvas.split()[3])
    base_blended = np.clip((np.array(base_frame).astype(np.float32) / 255.0) * (np.array(lines_only).astype(np.float32) / 255.0) * 255.0, 0, 255).astype(np.uint8)

    for step in range(15):
        opacity_factor = 1.0 - (step / 15.0)
        
        frame_img = Image.fromarray(base_blended).convert("RGBA")
        
        draft_np = np.array(draft_canvas)
        draft_np[:, :, 3] = (draft_np[:, :, 3].astype(np.float32) * opacity_factor).astype(np.uint8)
        faded_draft = Image.fromarray(draft_np)
        
        frame_img.paste(faded_draft, (0, 0), mask=faded_draft.split()[3])
        yield frame_img.convert("RGB")
        faded_draft.close()

    # Final frame (clean, guidelines completely erased)
    yield Image.fromarray(base_blended)

    paper_base.close()
    draft_canvas.close()
    drawing_canvas.close()
    filled_canvas.close()
