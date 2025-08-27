import numpy as np
import cv2

def log_LUT(ip_range, op_range=256, tol=0.04):
    """
    Build a log-like LUT with a small linear prefix to avoid log singularity.
    - ip_range: number of possible input values (e.g., max(pixel)+1)
    - op_range: number of output levels (e.g., 4096 for 12-bit)
    - tol: fraction of the output range reserved for the linear prefix
    Returns: 1D LUT of length ip_range (dtype=float32)
    """
    lut = np.zeros(ip_range, dtype=np.float32)
    lin_out = int(np.ceil(tol * op_range))   # linear prefix in output
    lin_in = int(np.ceil(tol * ip_range))   # linear prefix in input

    # Handle degenerate cases
    if ip_range <= 1:
        return lut

    # Linear prefix [0 .. lin_in-1] → [0 .. lin_out-1]
    if lin_in < 2:
        lut[0] = 0.0
    else:
        # Use linspace for exact length matching (avoids broadcast issues)
        lut[0:lin_in] = np.linspace(0, lin_out - 1, lin_in)

    # Log segment [lin_in .. ip_range-1] → [lin_out .. op_range]
    log_min = np.log(lin_in + 1)
    log_max = np.log(ip_range)
    k = (op_range - lin_out) / (log_max - log_min)
    idx = np.arange(lin_in, ip_range)
    lut[lin_in:ip_range] = lin_out + k * (np.log(idx) - log_min)

    # Ensure the very last entry hits the upper bound
    lut[-1] = float(op_range)
    return lut

def _bilinear_cdf_interpolate(cdfs, img_int, H, W, th, tw, nr, nc):
    """
    Bilinear interpolation of per-tile CDFs at every pixel.
    Returns values in [0, 1].
    """
    yi, xi = np.indices((H, W))

    # Tile indices for each pixel
    ry = np.clip(yi // th, 0, nr - 1)
    rx = np.clip(xi // tw, 0, nc - 1)

    # Relative offsets inside a tile [0,1]
    dy = (yi - ry * th) / th
    dx = (xi - rx * tw) / tw

    # Neighboring tiles
    ry1 = np.clip(ry + 1, 0, nr - 1)
    rx1 = np.clip(rx + 1, 0, nc - 1)

    # Bilinear weights
    w_tl = (1 - dy) * (1 - dx)
    w_tr = (1 - dy) * dx
    w_bl = dy * (1 - dx)
    w_br = dy * dx

    val = img_int  # intensity indices [0..nbins-1]

    # Interpolate CDF values
    out = (w_tl * cdfs[ry,  rx,  val] +
           w_tr * cdfs[ry,  rx1, val] +
           w_bl * cdfs[ry1, rx,  val] +
           w_br * cdfs[ry1, rx1, val])
    return out


def clahe_numpy_adaptive_clip(img, tile_grid=(8, 8), nbins=4096, alpha=1.0, beta=2.0):
    """
    CLAHE with variance-adaptive clip limit.
    - img: 16-bit grayscale image
    - tile_grid: (rows, cols) tiles (e.g., (8,8))
    - nbins: histogram bins for intensity quantization
    - alpha: base clip factor for low-variance tiles
    - beta: variance weighting factor
    """
    H, W = img.shape
    # Derive number of tiles; never below 8×8
    nr = max(8, H // 64)
    nc = max(8, W // 64)
    th, tw = int(np.ceil(H / nr)), int(np.ceil(W / nc))

    # Early exit for constant images
    vmin, vmax = img.min(), img.max()
    if vmax == vmin:
        return img.copy()

    # Quantize to [0..nbins-1] for bincount
    img_int = ((img - vmin) / (vmax - vmin) * (nbins - 1)).astype(np.uint16)

    # CDFs per tile
    cdfs = np.empty((nr, nc, nbins), np.float32)

    # Global variance (for adaptive clip)
    global_var = img.var()

    # Build CDF for each tile
    for r in range(nr):
        for c in range(nc):
            y0, y1 = r * th, min((r + 1) * th, H)
            x0, x1 = c * tw, min((c + 1) * tw, W)

            tile_vals = img_int[y0:y1, x0:x1].ravel()
            tile_orig = img[y0:y1, x0:x1]
            tile_var = tile_orig.var()

            # Adaptive clip limit (higher in complex regions)
            local_clip = alpha + beta * (tile_var / (global_var + 1e-8))
            max_per_bin = local_clip * tile_vals.size / nbins

            hist = np.bincount(tile_vals, minlength=nbins).astype(np.float32)
            excess = np.maximum(hist - max_per_bin, 0).sum()
            hist = np.minimum(hist, max_per_bin)
            hist += excess / nbins  # redistribute excess

            cdfs[r, c] = np.cumsum(hist) / tile_vals.size  # normalize to [0,1]

    # Bilinear interpolation of CDFs at pixel locations
    out = _bilinear_cdf_interpolate(cdfs, img_int, H, W, th, tw, nr, nc)

    # Map back to original dynamic range and dtype
    return (out * (vmax - vmin) + vmin).astype(img.dtype)

def guided_filter(I, p, r=4, eps=1e-4):
    """
    Edge-preserving guided filter.
    - I: guidance image
    - p: input to be filtered
    - r: radius of square window (size = 2r+1)
    - eps: regularization (avoids division by zero)
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    ones = np.ones_like(I, np.float32)

    def box(x):
        # Local sum in (2r+1)×(2r+1); reflections at borders
        return cv2.boxFilter(x, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1),
                             borderType=cv2.BORDER_REFLECT)

    N = box(ones)  # pixels per window

    mean_I = box(I) / N
    mean_p = box(p) / N
    corr_I = box(I * I) / N
    corr_Ip = box(I * p) / N

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    # Linear model p ≈ a*I + b within each window
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a) / N
    mean_b = box(b) / N

    return mean_a * I + mean_b

def clahe_guided(L, clip=2.0, grid=(8, 8), r=4, eps=1e-4):
    """
    One-level: CLAHE (with adaptive clip) - guided filter.
    - L: 16-bit grayscale image (luminance-like)
    - clip: not used directly here (kept for API symmetry)
    - grid: CLAHE tile grid
    - r, eps: guided filter params
    """
    global_std = np.std(L)

    # --- Adaptive α and β (innovation) ---
    # α (base clip limit): higher on globally flat images to allow stronger contrast.
    #   Decreases with global_std - avoids over-amplifying already high-contrast images.
    alpha_adapt = np.clip(2.0 - global_std / 500.0, 0.8, 2.0)

    # β (variance sensitivity): scales how much local variance raises the clip limit per tile.
    #   Increases with global_std - in globally complex images, allow tiles to push clip higher.
    beta_adapt = np.clip(global_std / 100.0, 0.8, 3.0)

    # CLAHE with adaptive clip; then edge-preserving smoothing with guidance = original
    L_cla = clahe_numpy_adaptive_clip(L, tile_grid=grid, nbins=4096, alpha=alpha_adapt, beta=beta_adapt)
    L_gf = guided_filter(I=L, p=L_cla, r=r, eps=eps)

    return L_gf.astype(np.float32)