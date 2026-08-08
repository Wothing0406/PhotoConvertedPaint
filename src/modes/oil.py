"""
modes/oil.py — Oil Painting (5-Layer Progressive Drawing)
==========================================================
Approach: TRUE master oil painting flow in 5 sequential stages
  - Layer 1: Structural Draft (Phác thảo cấu trúc):
    * Faint layout sketch lines (oval, axes, eyes) based on Gemini landmarks.
  - Layer 2: Blocking In (Tô màu khối lớn):
    * Large block strokes to define colors and shapes.
  - Layer 3: Medium Brushwork (Nét cọ trung):
    * Mid-sized strokes to add forms and structural values.
  - Layer 4: Fine Details & Highlights (Tả chi tiết ngũ quan):
    * Small, precise strokes focused on the face and details.
    * Blending original details back in portrait regions.
  - Layer 5: Eraser Phase & Impasto (Xoá nét nháp & Tạo nổi 3D):
    * Progressively fade out structural sketch guidelines.
    * Apply 3D Impasto shading and linen texture.
"""

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw

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
    **kw,
):
    w, h   = pil_img.size
    img_np = _boost_saturation(np.array(pil_img.convert("RGB")), 1.25)
    heightmap_np = np.zeros((h, w), dtype=np.float32)

    # ── Parse Gemini landmarks ───────────────────────────────────────────────
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

    canvas_np = np.full((h, w, 3), 242, dtype=np.uint8)
    
    # Yield initial wash + draft
    frame_img = Image.fromarray(canvas_np).convert("RGBA")
    frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
    yield frame_img.convert("RGB")

    gray          = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    saliency_norm = gpu_saliency(gray)
    gx_ang, gy_ang = gpu_sobel_gradients(gray, blur_k=9)

    max_dim    = max(w, h)
    passes_cfg = [
        # pass_idx, smooth_k, base_step, base_hl, pass_opacity
        (0, 31, max(16, max_dim // 24), max(22, max_dim // 18), 0.80), # Stage 2: Blocking
        (1, 15, max(10, max_dim // 40), max(12, max_dim // 30), 0.85), # Stage 3: Medium Form
        (2, 5,  max(5,  max_dim // 80), max(6,  max_dim // 60), 0.90), # Stage 4: Details
    ]

    frames_per_stage = target_frames // 5

    # ── Stage 2, 3, 4: Paint brushwork ────────────────────────────────────────
    for pass_idx, smooth_k, base_step, base_hl, pass_opacity in passes_cfg:
        color_src = cv2.GaussianBlur(img_np, (smooth_k, smooth_k), 0)
        strokes = []
        step = base_step
        
        for cy_s in range(step // 2, h, step):
            for cx_s in range(step // 2, w, step):
                rx = max(0, min(w - 1, int(cx_s + random.uniform(-step * 0.3, step * 0.3))))
                ry = max(0, min(h - 1, int(cy_s + random.uniform(-step * 0.3, step * 0.3))))
                sal = float(saliency_norm[ry, rx])
                
                dist = np.sqrt((rx - fc[0])**2 + (ry - fc[1])**2)

                if pass_idx == 2:
                    hl = max(2, int(base_hl * (1.0 - sal * 0.7)))
                elif pass_idx == 1:
                    hl = max(4, int(base_hl * (1.0 - sal * 0.5)))
                else:
                    hl = max(8, int(base_hl * (1.0 - sal * 0.3)))
                
                # Simplify background brushwork (larger, looser strokes)
                if dist >= R:
                    hl = int(hl * 1.5)
                
                hw = max(1, int(hl * 0.28))
                color = tuple(int(c) for c in color_src[ry, rx])
                v_x, v_y = gx_ang[ry, rx], gy_ang[ry, rx]
                
                mag = (v_x**2 + v_y**2)**0.5
                angle_deg = (int(np.degrees(np.arctan2(v_y, v_x) + np.pi / 2)) % 180
                             if mag > 0.1 else random.randint(0, 180))
                
                strokes.append((rx, ry, color, angle_deg, hl, hw, sal))

        if pass_idx == 2:
            strokes.sort(key=lambda s: s[6], reverse=True)
        else:
            random.shuffle(strokes)

        n_s = len(strokes)
        eff_s = max(1, n_s // max(10, frames_per_stage))

        for idx, (rx, ry, color, angle_deg, hl, hw, _) in enumerate(strokes):
            color_bgr = (color[2], color[1], color[0])
            _paint_stroke_soft(canvas_np, heightmap_np, rx, ry, color_bgr, angle_deg, hl, hw, pass_opacity)
            
            if idx % eff_s == 0 or idx == n_s - 1:
                frame_img = Image.fromarray(cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB)).convert("RGBA")
                # overlay draft guidelines
                frame_img.paste(draft_canvas, (0, 0), mask=draft_canvas.split()[3])
                yield frame_img.convert("RGB")

    # ── Stage 5: Eraser Phase (Guidelines fade out) & Final Shading ───────────
    final_rgb = cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB)
    shaded_np = _apply_impasto_lighting(final_rgb, heightmap_np, intensity=35.0)

    if saliency_norm is not None:
        detail_mask = np.clip((saliency_norm - 0.35) / 0.30, 0.0, 1.0)
        detail_mask = cv2.GaussianBlur(detail_mask, (15, 15), 0)[:, :, np.newaxis]
        orig_smooth = cv2.bilateralFilter(img_np, d=9, sigmaColor=50, sigmaSpace=50)
        shaded_np = (shaded_np.astype(np.float32) * (1.0 - detail_mask * 0.70) + 
                     orig_smooth.astype(np.float32) * (detail_mask * 0.70)).astype(np.uint8)

    # Eraser loop
    for step in range(15):
        opacity_factor = 1.0 - (step / 15.0)
        frame_img = Image.fromarray(shaded_np).convert("RGBA")
        
        draft_np = np.array(draft_canvas)
        draft_np[:, :, 3] = (draft_np[:, :, 3].astype(np.float32) * opacity_factor).astype(np.uint8)
        faded_draft = Image.fromarray(draft_np)
        
        frame_img.paste(faded_draft, (0, 0), mask=faded_draft.split()[3])
        yield frame_img.convert("RGB")
        faded_draft.close()

    # Blend soft canvas texture
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

    draft_canvas.close()
