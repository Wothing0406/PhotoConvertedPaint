"""
modes/oil.py — Master Oil Painting (Fixed)
============================================
Fix: gpu_bilateral approximation broke underpainting → revert to real
cv2.bilateralFilter (edge-preserving, correct). GPU used only for:
  - Saliency map      → gpu_saliency
  - Sobel gradients   → gpu_sobel_gradients (blur on GPU)
  - Soft-light blend  → gpu_soft_light
  - Canvas texture    → gpu_canvas_texture
"""

import cv2
import numpy as np
import random
from PIL import Image

from src.gpu_utils import (
    gpu_saliency,
    gpu_sobel_gradients,
    gpu_soft_light,
    gpu_canvas_texture,
    GPU_AVAILABLE,
)


def _paint_stroke_soft(canvas_np, cx, cy, color_bgr, angle_deg, hl, hw, opacity=0.85):
    h, w, _ = canvas_np.shape
    r = max(hl, hw) + 4
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w - 1, cx + r), min(h - 1, cy + r)
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return
    local_canvas = canvas_np[y1:y2, x1:x2]
    lh, lw, _    = local_canvas.shape
    stroke_mask  = np.zeros((lh, lw), dtype=np.uint8)
    lcx, lcy     = cx - x1, cy - y1
    cv2.ellipse(stroke_mask, (lcx, lcy), (max(1, hl), max(1, hw)),
                angle_deg, 0, 360, 255, -1, cv2.LINE_AA)
    blur_k      = max(3, int(min(hl, hw) * 0.4) | 1)
    stroke_mask = cv2.GaussianBlur(stroke_mask, (blur_k, blur_k), 0)
    jv = 14
    b     = max(0, min(255, color_bgr[0] + random.randint(-jv, jv)))
    g     = max(0, min(255, color_bgr[1] + random.randint(-jv, jv)))
    r_val = max(0, min(255, color_bgr[2] + random.randint(-jv, jv)))
    stroke_color = np.array([b, g, r_val], dtype=np.float32)
    mask_f = (stroke_mask.astype(np.float32) / 255.0) * opacity
    mask_f = np.expand_dims(mask_f, axis=2)
    local_canvas[:] = (
        local_canvas.astype(np.float32) * (1.0 - mask_f) + stroke_color * mask_f
    ).astype(np.uint8)


def draw(
    pil_img: Image.Image,
    *,
    blur_size: int = 7,
    jitter: float = 0.65,
    batch_size: int = 20,
    **_kw,
):
    w, h   = pil_img.size
    img_np = np.array(pil_img.convert("RGB"))

    # ── Step 1: Real bilateral underpainting (edge-preserving, MUST be real) ──
    # NOTE: cv2.bilateralFilter is edge-preserving. GPU approx breaks this.
    smooth_under = cv2.bilateralFilter(img_np, d=15, sigmaColor=120, sigmaSpace=120)
    underpainting = cv2.medianBlur(smooth_under, 9)
    canvas_np     = cv2.cvtColor(underpainting, cv2.COLOR_RGB2BGR)

    pf = Image.fromarray(cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB))
    yield pf.copy(); pf.close()

    # ── Step 2: Saliency map (GPU) ────────────────────────────────────────────
    gray          = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    saliency_norm = gpu_saliency(gray)

    # ── Step 3: Directional Sobel gradients (blur GPU, Sobel CPU) ────────────
    gx_ang, gy_ang = gpu_sobel_gradients(gray, blur_k=9)

    # ── Step 4: Multi-pass Brush Strokes ──────────────────────────────────────
    max_dim    = max(w, h)
    passes_cfg = [
        (31, max(24, max_dim // 20), max(22, max_dim // 18), 0.70),
        (15, max(12, max_dim // 36), max(12, max_dim // 32), 0.80),
        (5,  max(6,  max_dim // 75), max(5,  max_dim // 68), 0.90),
    ]
    cx0, cy0 = w / 2.0, h / 2.0

    for pass_idx, (smooth_k, base_step, base_hl, pass_opacity) in enumerate(passes_cfg):
        color_src = cv2.GaussianBlur(img_np, (smooth_k, smooth_k), 0)
        strokes   = []
        step = base_step
        for cy_s in range(step // 2, h, step):
            for cx_s in range(step // 2, w, step):
                rx = max(0, min(w - 1, int(cx_s + random.uniform(-step * 0.3, step * 0.3))))
                ry = max(0, min(h - 1, int(cy_s + random.uniform(-step * 0.3, step * 0.3))))
                sal = float(saliency_norm[ry, rx])
                if pass_idx == 2:
                    hl = max(2, int(base_hl * (1.0 - sal * 0.7)))
                elif pass_idx == 1:
                    hl = max(4, int(base_hl * (1.0 - sal * 0.5)))
                else:
                    hl = max(8, int(base_hl * (1.0 - sal * 0.3)))
                hw    = max(1, int(hl * 0.28))
                color = tuple(int(c) for c in color_src[ry, rx])
                v_x, v_y = gx_ang[ry, rx], gy_ang[ry, rx]
                mag = (v_x**2 + v_y**2)**0.5
                angle_deg = (int(np.degrees(np.arctan2(v_y, v_x) + np.pi / 2)) % 180
                             if mag > 1.5 else random.randint(0, 180))
                strokes.append((rx, ry, color, angle_deg, hl, hw, sal))

        strokes.sort(key=lambda s: (s[6], -((s[0] - cx0)**2 + (s[1] - cy0)**2)))
        eff_batch = max(batch_size, len(strokes) // 100)
        for i, (cx_s, cy_s, color, ang, hl, hw, _) in enumerate(strokes):
            _paint_stroke_soft(canvas_np, cx_s, cy_s,
                               (color[2], color[1], color[0]), ang, hl, hw, pass_opacity)
            if i % eff_batch == 0 or i == len(strokes) - 1:
                pf = Image.fromarray(cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB))
                yield pf.copy(); pf.close()

    # ── Step 5: Soft-Light Canvas Texture Overlay (GPU) ───────────────────────
    canvas_tex = gpu_canvas_texture(w, h)
    canvas_rgb = cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB)
    canvas_f   = canvas_rgb.astype(np.float32) / 255.0
    tex_f      = canvas_tex.astype(np.float32) / 255.0
    tex_f_3ch  = np.stack([tex_f, tex_f, tex_f], axis=2)
    final_f    = gpu_soft_light(canvas_f, tex_f_3ch)
    final_rgb  = (np.clip(final_f, 0.0, 1.0) * 255).astype(np.uint8)
    pil_final  = Image.fromarray(final_rgb)
    yield pil_final.copy()
    pil_final.close()
