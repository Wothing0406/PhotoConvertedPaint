"""
modes/blueprint.py — Paint-by-Numbers Blueprint (Optimized for flat cartoon blocks)
====================================================================================
Fix: Starry Night / Van Gogh textures produce thousands of tiny noise segments.
We must aggressively merge regions and blur the label map to produce broad,
clean flat areas suitable for a real paint-by-numbers canvas.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw


def _kmeans_seg(img_rgb: np.ndarray, k: int = 12):
    h, w = img_rgb.shape[:2]
    # Smooth heavily to merge small details
    smooth = cv2.bilateralFilter(img_rgb, d=15, sigmaColor=120, sigmaSpace=120)
    smooth = cv2.medianBlur(smooth, 11)
    
    pixels = smooth.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    
    centers = np.uint8(centers)
    seg = labels.reshape(h, w).astype(np.uint8)
    
    # Run consecutive median blurs to remove speckles and straighten region boundaries
    seg = cv2.medianBlur(seg, 13)
    seg = cv2.medianBlur(seg, 9)
    return seg, centers


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
    batch_size: int = 20,
    **_kw,
):
    """Yields progressive PIL frames of a paint-by-numbers blueprint."""
    w, h = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # Reduce color count to 12 for clean, easily paintable regions
    k = 12
    seg, centers = _kmeans_seg(img_np, k=k)

    canvas = Image.new("RGB", (w, h), (252, 250, 244))
    draw_layer = ImageDraw.Draw(canvas)
    yield canvas.copy()

    # ── Pass 1: Fill color regions ───────────────────────────────────────────
    region_areas = [(ki, np.sum(seg == ki)) for ki in range(k)]
    region_areas.sort(key=lambda x: -x[1])

    for ki, area in region_areas:
        if area < 600:  # Increase noise threshold significantly (was 150)
            continue
            
        mask = ((seg == ki).astype(np.uint8) * 255)
        color = tuple(int(c) for c in centers[ki])
        
        # Clean mask from small holes before contour extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_drawn = False
        for cnt in cnts:
            if cv2.contourArea(cnt) < 500:  # Filter out small islands
                continue
            p = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, max(0.2, 0.004 * p), True)
            pts = [tuple(pt[0]) for pt in approx]
            if len(pts) >= 3:
                draw_layer.polygon(pts, fill=color)
                has_drawn = True
                
        if has_drawn:
            yield canvas.copy()

    # ── Pass 2: Clean boundaries ─────────────────────────────────────────────
    boundary_mask = _region_boundaries(seg)
    # Erode boundary mask slightly so boundaries are thin
    cnts_b, _ = cv2.findContours(boundary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_b = sorted(cnts_b, key=lambda c: cv2.arcLength(c, False), reverse=True)
    
    n = len(cnts_b)
    eff_batch = max(batch_size, n // 60) if n > 0 else batch_size

    for idx, cnt in enumerate(cnts_b):
        p = cv2.arcLength(cnt, False)
        if p < 25:  # Ignore short segments (was 8)
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
            draw_layer.line(result, fill=(60, 60, 60), width=1)
            
        if idx % eff_batch == 0 or idx == n - 1:
            yield canvas.copy()

    # ── Pass 3: Placing number labels ────────────────────────────────────────
    for ki in range(k):
        mask = (seg == ki).astype(np.uint8) * 255
        cnts_lbl, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_lbl:
            continue
        # Label only moderately large regions
        for cnt in cnts_lbl:
            if cv2.contourArea(cnt) < 2000:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            lx = int(M["m10"] / M["m00"])
            ly = int(M["m01"] / M["m00"])
            
            # Color contrast for labels
            bg = tuple(int(c) for c in centers[ki])
            lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
            txt_col = (40, 40, 40) if lum > 128 else (230, 230, 230)
            draw_layer.text((lx - 4, ly - 6), str(ki + 1), fill=txt_col)

    yield canvas.copy()
