"""
modes/oil.py — Oil Painting
============================
Approach: TRUE master oil painting flow
  1. Start with a rough, light gray canvas (gesso texture ground).
  2. Pass 1: "Blocking In" — progressively draw large block strokes to define colors and shapes.
  3. Pass 2: "Medium brushwork" — draw mid-sized strokes to add forms and structural values.
  4. Pass 3: "Fine details & highlights" — draw small strokes to refine details, eyes, edges.
  5. Pass 4: Apply 3D Impasto lighting and canvas texture to make the paint look thick and rich.
  6. Pass 5: Detail Restoration on highly salient regions (face, eyes, text) using bilateral
     detail blending to preserve high-fidelity portrait details.
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


def _paint_stroke_soft(canvas_np, heightmap_np, cx, cy, color_bgr, angle_deg, hl, hw, opacity=0.85):
    h, w, _ = canvas_np.shape
    r = max(hl, hw) + 4
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w - 1, cx + r), min(h - 1, cy + r)
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return
    local_canvas = canvas_np[y1:y2, x1:x2]
    local_height = heightmap_np[y1:y2, x1:x2]
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
    
    # Accumulate height for 3D impasto
    local_height[:] = np.maximum(local_height, mask_f * 255.0)
    
    mask_f_3d = np.expand_dims(mask_f, axis=2)
    local_canvas[:] = (
        local_canvas.astype(np.float32) * (1.0 - mask_f_3d) + stroke_color * mask_f_3d
    ).astype(np.uint8)


def _apply_impasto_lighting(color_img, heightmap, intensity=35.0):
    heightmap_smooth = cv2.GaussianBlur(heightmap, (5, 5), 0)
    dx = cv2.Sobel(heightmap_smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(heightmap_smooth, cv2.CV_32F, 0, 1, ksize=3)
    
    normal_x = -dx
    normal_y = -dy
    normal_z = np.full_like(dx, intensity)
    
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2 + 1e-7)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    
    light_dir = np.array([-1.0, -1.0, 1.2])
    light_dir /= np.sqrt(np.sum(light_dir**2))
    
    diffuse = normal_x * light_dir[0] + normal_y * light_dir[1] + normal_z * light_dir[2]
    shading = 0.88 + 0.22 * np.clip(diffuse, 0.0, 1.0)
    
    shaded_img = np.clip(color_img.astype(np.float32) * shading[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return shaded_img


def _boost_saturation(img_rgb, factor=1.25):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def draw(
    pil_img: Image.Image,
    *,
    blur_size: int = 7,
    jitter: float = 0.65,
    batch_size: int = 20,
    **_kw,
):
    w, h   = pil_img.size
    img_np = _boost_saturation(np.array(pil_img.convert("RGB")), 1.25)
    heightmap_np = np.zeros((h, w), dtype=np.float32)

    # ── Step 1: Canvas starts as gesso gray textured ground ──────────────────
    canvas_np = np.full((h, w, 3), 242, dtype=np.uint8)
    yield Image.fromarray(canvas_np).convert("RGB")

    gray          = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    saliency_norm = gpu_saliency(gray)
    gx_ang, gy_ang = gpu_sobel_gradients(gray, blur_k=9)

    max_dim    = max(w, h)
    passes_cfg = [
        # pass_idx, smooth_k, base_step, base_hl, pass_opacity
        (0, 31, max(16, max_dim // 24), max(22, max_dim // 18), 0.80), # Blocking
        (1, 15, max(10, max_dim // 40), max(12, max_dim // 30), 0.85), # Medium Form
        (2, 5,  max(5,  max_dim // 80), max(6,  max_dim // 60), 0.90), # Details
    ]

    for pass_idx, smooth_k, base_step, base_hl, pass_opacity in passes_cfg:
        color_src = cv2.GaussianBlur(img_np, (smooth_k, smooth_k), 0)
        strokes = []
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
                
                hw = max(1, int(hl * 0.28))
                color = tuple(int(c) for c in color_src[ry, rx])
                v_x, v_y = gx_ang[ry, rx], gy_ang[ry, rx]
                
                mag = (v_x**2 + v_y**2)**0.5
                angle_deg = (int(np.degrees(np.arctan2(v_y, v_x) + np.pi / 2)) % 180
                             if mag > 0.1 else random.randint(0, 180))
                
                strokes.append((rx, ry, color, angle_deg, hl, hw, sal))

        # Paint fine details on subject first
        if pass_idx == 2:
            strokes.sort(key=lambda s: s[6], reverse=True)
        else:
            random.shuffle(strokes)

        n_s = len(strokes)
        eff_s = max(1, min(batch_size, max(1, n_s // 80)))

        for idx, (rx, ry, color, angle_deg, hl, hw, _) in enumerate(strokes):
            color_bgr = (color[2], color[1], color[0])
            _paint_stroke_soft(canvas_np, heightmap_np, rx, ry, color_bgr, angle_deg, hl, hw, pass_opacity)
            
            if idx % eff_s == 0 or idx == n_s - 1:
                yield Image.fromarray(cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB))

    # ── Pass 4: Apply 3D Impasto paint height lighting ───────────────────────
    final_rgb = cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB)
    shaded_np = _apply_impasto_lighting(final_rgb, heightmap_np, intensity=35.0)

    # ── Pass 5: Detail Restoration on highly salient regions ─────────────────
    if saliency_norm is not None:
        # Create a smooth blending mask for salient regions (face/eyes/hands)
        detail_mask = np.clip((saliency_norm - 0.35) / 0.30, 0.0, 1.0)
        detail_mask = cv2.GaussianBlur(detail_mask, (15, 15), 0)[:, :, np.newaxis]
        
        # Smooth bilateral details of original
        orig_smooth = cv2.bilateralFilter(img_np, d=9, sigmaColor=50, sigmaSpace=50)
        
        # Blend original details back on salient regions to preserve high-fidelity portrait features
        shaded_np = (shaded_np.astype(np.float32) * (1.0 - detail_mask * 0.70) + 
                     orig_smooth.astype(np.float32) * (detail_mask * 0.70)).astype(np.uint8)

    # Blend soft canvas texture at the end
    try:
        tex = gpu_canvas_texture(w, h)
        shaded_f = shaded_np.astype(np.float32) / 255.0
        tex_f = tex.astype(np.float32)[:, :, np.newaxis] / 255.0
        blended = gpu_soft_light(shaded_f, np.repeat(tex_f, 3, axis=2))
        final_img = Image.fromarray((np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8))
        yield final_img.copy()
        final_img.close()
    except Exception as te:
        print(f"[oil] Canvas texture failed: {te}")
        yield Image.fromarray(shaded_np)
