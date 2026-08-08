"""
modes/blueprint.py — Paint-by-Numbers Blueprint
=================================================
Approach: TRUE paint-by-numbers painting flow
  1. Start with a warm paper canvas.
  2. Draw all region outlines (black/dark-gray contours) and centroid color labels (numbers).
     This yields the complete blank coloring template first.
  3. Progressively fill color region-by-region (largest to smallest) to simulate the painting process.
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
    **_kw,
):
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # Region segmentation
    k = 12
    seg, centers = _kmeans_seg(img_np, k=k)

    # ── Canvas starts as warm paper template ─────────────────────────────────
    canvas = Image.new("RGB", (w, h), (252, 250, 244))
    draw_layer = ImageDraw.Draw(canvas)
    yield canvas.copy()

    # ── Step 1: Draw All Boundaries (Template lines) ──────────────────────────
    boundary_mask = _region_boundaries(seg)
    thinned_b = _thin_edges(boundary_mask)
    cnts_b, _ = cv2.findContours(thinned_b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_b = sorted(cnts_b, key=lambda c: cv2.arcLength(c, False), reverse=True)
    
    n_b = len(cnts_b)
    eff_b = max(1, min(batch_size, max(1, n_b // 60)))

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
            draw_layer.line(result, fill=(90, 90, 90), width=1)
            
        if idx % eff_b == 0 or idx == n_b - 1:
            yield canvas.copy()

    # ── Step 2: Draw Labels (Numbers) ─────────────────────────────────────────
    # Keep number labels on top of regions
    labels_to_place = []
    for ki in range(k):
        mask = (seg == ki).astype(np.uint8) * 255
        kernel_lbl = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_lbl)
        cnts_lbl, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in cnts_lbl:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                labels_to_place.append((cx, cy, str(ki + 1)))

    for cx, cy, txt in labels_to_place:
        # Draw small dark gray label text
        draw_layer.text((cx - 4, cy - 6), txt, fill=(80, 80, 80))
    yield canvas.copy()

    # Save the template states
    template_canvas = canvas.copy()

    # ── Step 3: Color painting process (Fill regions largest to smallest) ────
    region_areas = []
    for ki in range(k):
        mask = ((seg == ki).astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts_reg, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_reg:
            area = cv2.contourArea(cnt)
            if area >= 200:
                region_areas.append((ki, cnt, area))

    region_areas.sort(key=lambda x: -x[2])  # Paint large shapes first

    # Paint regions progressively
    n_r = len(region_areas)
    eff_r = max(1, min(batch_size, max(1, n_r // 80)))

    for idx, (ki, cnt, area) in enumerate(region_areas):
        p = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(0.2, 0.004 * p), True)
        pts = [tuple(pt[0]) for pt in approx]
        if len(pts) >= 3:
            color = tuple(int(c) for c in centers[ki])
            # Draw color region
            draw_layer.polygon(pts, fill=color)
            
        if idx % eff_r == 0 or idx == n_r - 1:
            # Composite template outlines and labels back on top so they remain visible
            colored_np = np.array(canvas)
            template_np = np.array(template_canvas)
            # Lines are drawn on top using a simple minimum blend
            blended = np.minimum(colored_np, template_np)
            yield Image.fromarray(blended)

    # Yield final completely colored image with lines
    yield Image.fromarray(np.minimum(np.array(canvas), np.array(template_canvas)))
    canvas.close()
    template_canvas.close()
