import numpy as np
import cv2

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


