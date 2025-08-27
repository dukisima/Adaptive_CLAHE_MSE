import cv2
import matplotlib.pyplot as plt
import numpy as np

import filters 
import metrics
import pyramids

# 1) Load + log compression (no autocrop; images already cropped)
image_data = cv2.imread("/Users/macbookair/Code/Adaptive_CLAHE_MSE/Adaptive_CLAHE_MSE/data/raw/CHEST_PA_N_v2.tiff", cv2.IMREAD_UNCHANGED)
image = image_data  # already cropped

ip_range = int(image.max() + 1)
logLUT = filters.log_LUT(ip_range, op_range=4096, tol=0.001)  # float32 LUT
image_enhanced = logLUT[image]
image_input = image_enhanced.astype(np.float32)

# 2) Laplacian pyramid
N = 6
LPyr, GPyr, Res, size_vec = pyramids.im_pyr_decomp(image_input, N)

# 3) Process levels 2–5 with CLAHE+Guided
for level in [2, 3, 4, 5]:
    L = LPyr[level].astype(np.float32)
    if np.ptp(L) < 1e-5:  # skip near-constant levels
        continue
    L_mod = filters.clahe_guided(L, clip=2.8, grid=(18, 18), r=4, eps=1e-4)
    LPyr[level] = L_mod

# 4) Reconstruct
im_rec = pyramids.im_pyr_recon(LPyr, Res, size_vec)

# 5) Metrics
metrics.sharpness_metrics(image_input, im_rec)

# 6) Display (negative rendering like in your note)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(image_input.max() - image_input, cmap='gray')
ax[0].set_title("Original"); ax[0].axis('off')
ax[1].imshow(image_input.max() - im_rec, cmap='gray')
ax[1].set_title("CLAHE + Guided"); ax[1].axis('off')
plt.tight_layout()
plt.show()