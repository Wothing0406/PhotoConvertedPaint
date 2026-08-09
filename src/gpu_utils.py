"""
src/gpu_utils.py — Centralised GPU Acceleration Layer
======================================================
All 5 drawing modes route through this module.

Priority order for GPU operations:
  1. CuPy  (cupyx.scipy.ndimage for convolutions)   — RTX 3050 6GB VRAM
  2. NumPy CPU fallback                               — always safe

Usage:
    from src.gpu_utils import gpu_gaussian_blur, GPU_AVAILABLE

All functions accept NumPy arrays and return NumPy arrays.
"""

import numpy as np
import cv2

# ── 1. Probe CuPy ─────────────────────────────────────────────────────────────
GPU_AVAILABLE = False
_cp = None
_cpnd = None   # cupyx.scipy.ndimage

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd

    _test = cp.zeros((4, 4), dtype=cp.float32)
    del _test

    _cp   = cp
    _cpnd = cpnd
    GPU_AVAILABLE = True
    _dev_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    _vram_mb  = cp.cuda.Device(0).mem_info[1] // 1024**2
    print(f"[GPU] CuPy {cp.__version__} ready — {_dev_name} ({_vram_mb} MB VRAM)")
except Exception as e:
    print(f"[GPU] CuPy unavailable ({e}), using CPU fallback.")


# ── 2. Gaussian Blur ──────────────────────────────────────────────────────────
def gpu_gaussian_blur(img_np: np.ndarray, ksize_or_sigma, *, sigma: float = 0.0) -> np.ndarray:
    if isinstance(ksize_or_sigma, int):
        k = max(3, ksize_or_sigma | 1)
        sig = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    else:
        sig = float(ksize_or_sigma)
        k = int(6 * sig + 1) | 1

    if GPU_AVAILABLE:
        try:
            g = _cp.asarray(img_np.astype(np.float32))
            if g.ndim == 2:
                out = _cpnd.gaussian_filter(g, sigma=sig)
            else:
                channels = [_cpnd.gaussian_filter(g[:, :, c], sigma=sig) for c in range(g.shape[2])]
                out = _cp.stack(channels, axis=2)
            result = _cp.asnumpy(out)
            return (np.clip(result, 0, 255).astype(np.uint8)
                    if img_np.dtype == np.uint8 else result.astype(img_np.dtype))
        except Exception:
            pass
    return cv2.GaussianBlur(img_np, (k, k), sig)


# ── 3. Bilateral Filter ────────────────────────────────────────────────────────
def gpu_bilateral(img_np: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    if GPU_AVAILABLE:
        try:
            img_f = _cp.asarray(img_np.astype(np.float32))
            sig_s = sigma_space / 3.0
            blurred = _cp.stack(
                [_cpnd.gaussian_filter(img_f[:, :, c], sigma=sig_s) for c in range(3)], axis=2
            )
            lum_o = 0.299*img_f[:,:,0] + 0.587*img_f[:,:,1] + 0.114*img_f[:,:,2]
            lum_b = 0.299*blurred[:,:,0] + 0.587*blurred[:,:,1] + 0.114*blurred[:,:,2]
            w = _cp.exp(-((lum_o - lum_b)**2) / (2*sigma_color**2))[:, :, _cp.newaxis]
            result = w * blurred + (1-w) * img_f
            return _cp.asnumpy(_cp.clip(result, 0, 255).astype(_cp.uint8))
        except Exception:
            pass
    return cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)


# ── 4. Canny ──────────────────────────────────────────────────────────────────
def gpu_canny(gray_np: np.ndarray, lo: float, hi: float, blur_k: int = 3) -> np.ndarray:
    blurred = gpu_gaussian_blur(gray_np, blur_k)
    return cv2.Canny(blurred, lo, hi)


# ── 5. Pencil Dodge Blend ─────────────────────────────────────────────────────
def gpu_pencil_dodge(gray_np: np.ndarray, blur_r: int) -> np.ndarray:
    # Fine scale for sharp details/contours
    b_fine = max(3, (blur_r * 4 + 1) | 1)
    sig_fine = 0.3 * ((b_fine - 1) * 0.5 - 1) + 0.8
    
    # Coarse scale for large graphite shading/cushioning (di chì / đồ chì nền)
    sig_coarse = max(15.0, sig_fine * 6.5)
    b_coarse = int(sig_coarse * 6 + 1) | 1

    if GPU_AVAILABLE:
        try:
            g = _cp.asarray(gray_np.astype(_cp.float32))
            inv = 255.0 - g
            
            # Fine dodge
            blur_inv_fine = _cpnd.gaussian_filter(inv, sigma=sig_fine)
            sketch_fine = _cp.where(blur_inv_fine >= 255, 255.0,
                                   _cp.clip(g / (1.0 - blur_inv_fine/255.0 + 1e-7), 0, 255))
            
            # Coarse dodge
            blur_inv_coarse = _cpnd.gaussian_filter(inv, sigma=sig_coarse)
            sketch_coarse = _cp.where(blur_inv_coarse >= 255, 255.0,
                                     _cp.clip(g / (1.0 - blur_inv_coarse/255.0 + 1e-7), 0, 255))
            
            # Blend: 45% fine details + 55% coarse shading (cushions large structures like walls/clothes)
            blended = 0.45 * sketch_fine + 0.55 * sketch_coarse
            return _cp.asnumpy(blended).astype(np.uint8)
        except Exception:
            pass

    inv = 255.0 - gray_np.astype(np.float32)
    # Fine
    blur_inv_fine = cv2.GaussianBlur(inv, (b_fine, b_fine), 0)
    sketch_fine = np.clip(
        np.where(blur_inv_fine >= 255, 255, gray_np.astype(np.float32) / (1.0 - blur_inv_fine/255.0 + 1e-7)),
        0, 255
    )
    # Coarse
    b_coarse_clamped = min(101, b_coarse | 1)
    blur_inv_coarse = cv2.GaussianBlur(inv, (b_coarse_clamped, b_coarse_clamped), 0)
    sketch_coarse = np.clip(
        np.where(blur_inv_coarse >= 255, 255, gray_np.astype(np.float32) / (1.0 - blur_inv_coarse/255.0 + 1e-7)),
        0, 255
    )
    
    blended = 0.45 * sketch_fine + 0.55 * sketch_coarse
    return blended.astype(np.uint8)


def gpu_pencil_dodge_channel(ch_np: np.ndarray, blur_r: int) -> np.ndarray:
    return gpu_pencil_dodge(ch_np, blur_r)


# ── 6. Soft-Light Blend ────────────────────────────────────────────────────────
def gpu_soft_light(base_f: np.ndarray, overlay_f: np.ndarray) -> np.ndarray:
    """Both inputs: float32 [0,1], same shape. Returns float32 [0,1]."""
    if GPU_AVAILABLE:
        try:
            b = _cp.asarray(base_f.astype(np.float32))
            o = _cp.asarray(overlay_f.astype(np.float32))
            result = _cp.where(
                o < 0.5,
                (1.0 - 2.0*o) * (b**2) + 2.0*o*b,
                (2.0*o - 1.0) * (_cp.sqrt(_cp.maximum(b, 0)) - b) + b
            )
            return _cp.asnumpy(_cp.clip(result, 0, 1)).astype(np.float32)
        except Exception:
            pass
    return np.clip(np.where(
        overlay_f < 0.5,
        (1.0-2*overlay_f)*(base_f**2)+2*overlay_f*base_f,
        (2*overlay_f-1)*(np.sqrt(np.maximum(base_f,0))-base_f)+base_f
    ), 0, 1).astype(np.float32)


# ── 7. Multiply Blend ─────────────────────────────────────────────────────────
def gpu_multiply(a_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    if GPU_AVAILABLE:
        try:
            return _cp.asnumpy(_cp.clip(_cp.asarray(a_f)*_cp.asarray(b_f), 0, 1)).astype(np.float32)
        except Exception:
            pass
    return np.clip(a_f * b_f, 0, 1).astype(np.float32)


# ── 8. Dark Mask Composite ────────────────────────────────────────────────────
def gpu_dark_mask_composite(canvas_f: np.ndarray, dark_mask: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """canvas_f: (H,W,3) float32; dark_mask: (H,W) float32 [0,1]. Returns (H,W,3) uint8."""
    if GPU_AVAILABLE:
        try:
            c = _cp.asarray(canvas_f)
            m = _cp.asarray(dark_mask[:, :, np.newaxis])
            result = _cp.clip(c*(1.0-m*strength) + 28.0*m*strength, 0, 255)
            return _cp.asnumpy(result).astype(np.uint8)
        except Exception:
            pass
    m3d = dark_mask[:, :, np.newaxis]
    return np.clip(canvas_f*(1.0-m3d*strength)+28.0*m3d*strength, 0, 255).astype(np.uint8)


# ── 9. Sobel Gradients ────────────────────────────────────────────────────────
def gpu_sobel_gradients(gray_np: np.ndarray, blur_k: int = 9):
    """Returns (gx, gy) both (H,W) float32."""
    blurred = gpu_gaussian_blur(gray_np, blur_k)
    gx = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)
    return gx, gy


# ── 10. Saliency Map ──────────────────────────────────────────────────────────
def gpu_saliency(gray_np: np.ndarray) -> np.ndarray:
    """
    Returns (H,W) float32 normalised [0,1].
    
    Strategy: Center-Weighted Saliency Map
      - Compute frequency-based saliency (DoG: Difference of Gaussians) for subject detection
      - Multiply with a Gaussian center-weight (subjects are typically in frame center)
      - This prevents flat-texture backgrounds (walls, tiles) from having high saliency
        even though they have many edges
    """
    h, w = gray_np.shape

    # Difference of Gaussians (DoG) — captures mid-frequency blobs = human subjects
    blurred_fine = cv2.GaussianBlur(gray_np.astype(np.float32), (5, 5), 1.0)
    blurred_coarse = cv2.GaussianBlur(gray_np.astype(np.float32), (51, 51), 15.0)
    dog = np.abs(blurred_fine - blurred_coarse)

    # Smooth DoG to get broad salient regions
    dog_smooth = cv2.GaussianBlur(dog, (31, 31), 10.0)

    # Center-weight Gaussian: subjects in center get priority over wall edges
    cy, cx = h / 2.0, w / 2.0
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    sigma_x = w * 0.40
    sigma_y = h * 0.42
    center_weight = np.exp(
        -((xg - cx)**2 / (2 * sigma_x**2) + (yg - cy)**2 / (2 * sigma_y**2))
    ).astype(np.float32)

    # Combine: DoG saliency × center weight
    sal = dog_smooth * center_weight

    # Also add a small edge gradient component for fine line details
    sx = cv2.Scharr(gray_np, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(gray_np, cv2.CV_32F, 0, 1)
    edge_mag = cv2.magnitude(sx, sy)
    edge_smooth = cv2.GaussianBlur(edge_mag, (21, 21), 6.0)
    edge_smooth *= center_weight  # gate edge saliency by center weight too

    # Combined: 70% DoG blob + 30% center-gated edges
    sal = sal * 0.70 + edge_smooth * 0.30

    s_max = float(sal.max()) or 1.0
    return (sal / s_max).astype(np.float32)




# ── 11. Canvas Noise Texture ──────────────────────────────────────────────────
def gpu_canvas_texture(w: int, h: int) -> np.ndarray:
    """Returns (H,W) uint8 linen canvas texture."""
    if GPU_AVAILABLE:
        try:
            rng = _cp.random.default_rng(42)
            noise = rng.integers(0, 255, (h, w), dtype=_cp.uint8).astype(_cp.float32)
            ht = _cpnd.gaussian_filter(noise, sigma=[0.5, 7.0])
            vt = _cpnd.gaussian_filter(noise, sigma=[7.0, 0.5])
            canvas = _cp.clip((ht + vt) / 2, 0, 255)
            lo, hi = float(canvas.min()), float(canvas.max())
            if hi > lo:
                canvas = (canvas - lo) / (hi - lo) * 255
            canvas = _cpnd.gaussian_filter(canvas, sigma=1.5)
            normalized = 128 + (canvas - 128) * 0.08
            return _cp.asnumpy(_cp.clip(normalized, 0, 255).astype(_cp.uint8))
        except Exception:
            pass
    noise = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    ht = cv2.GaussianBlur(noise, (15, 1), 0)
    vt = cv2.GaussianBlur(noise, (1, 15), 0)
    canvas = cv2.addWeighted(ht, 0.5, vt, 0.5, 0)
    canvas = cv2.equalizeHist(canvas)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return (128 + (canvas.astype(np.float32) - 128) * 0.08).astype(np.uint8)


# ── 12. XDoG ──────────────────────────────────────────────────────────────────
def gpu_xdog(gray_np: np.ndarray, sigma1: float, sigma2: float,
             tau: float = 0.98, phi: float = 10.0, epsilon: float = -0.1) -> np.ndarray:
    """XDoG edge map. Input (H,W) uint8. Output (H,W) uint8 binary."""
    if GPU_AVAILABLE:
        try:
            g = _cp.asarray(gray_np.astype(_cp.float32))
            g1 = _cpnd.gaussian_filter(g, sigma=sigma1)
            g2 = _cpnd.gaussian_filter(g, sigma=sigma2)
            diff = g1 - tau * g2
            result = _cp.where(diff >= epsilon, 1.0, 1.0+_cp.tanh(phi*(diff-epsilon)))
            ink = _cp.asnumpy((_cp.clip(result, 0, 1)*255).astype(_cp.uint8))
            _, binary = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return binary
        except Exception:
            pass
    gf = gray_np.astype(np.float32)
    g1 = cv2.GaussianBlur(gf, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gf, (0, 0), sigma2)
    diff = g1 - tau * g2
    result = np.where(diff >= epsilon, 1.0, 1.0+np.tanh(phi*(diff-epsilon)))
    ink = (np.clip(result, 0, 1)*255).astype(np.uint8)
    _, binary = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary


# ── Status ────────────────────────────────────────────────────────────────────
def gpu_status() -> str:
    if GPU_AVAILABLE and _cp is not None:
        free, total = _cp.cuda.Device(0).mem_info
        name = _cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        return f"GPU ON  — {name} | VRAM {free//1024**2}/{total//1024**2} MB free"
    return "GPU OFF — CPU fallback active"


def gpu_clear_cache():
    if GPU_AVAILABLE and _cp is not None:
        try:
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()
            print("[GPU] VRAM Memory Pool cleared successfully.")
        except Exception as e:
            print(f"[GPU] Error clearing VRAM cache: {e}")
