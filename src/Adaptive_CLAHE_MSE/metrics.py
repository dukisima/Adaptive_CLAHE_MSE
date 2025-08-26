from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
from skimage.util import view_as_windows
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import cv2
import pywt

def sharpness_metrics(original, enhanced):
    """
    Calculates and prints out contrast, sharpens and noise increase between original and enhanced image
    """
    def rms_contrast(img):
        return img.std()
    def tenengrad(img):
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        return np.mean(sobel_x ** 2 + sobel_y ** 2)

    def laplacian_variance(img):
        lap = lambda x: cv2.Laplacian(x.astype(np.float32), cv2.CV_32F).var()
        return lap(img)

    def michelson_contrast(img):
        img = img.astype(np.float32) #so that the return is between 0 and 1
        I_max = img.max()
        I_min = img.min()
        return (I_max - I_min) / (I_max + I_min + 1e-8)

    def entropy(img):
        return shannon_entropy(img)

    def ssim_index(img1, img2):
        img1_8 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img2_8 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return ssim(img1_8, img2_8)

    def estimate_sigma_noise(img):
      sigma_est_orig = estimate_sigma(img, channel_axis=None, average_sigmas=True)
      return sigma_est_orig

    def nois_estimate_sigma_dwt(a, n_res=1, wv_filt='db2'):
      cA = a.astype(float)
      ns = []
      ss = []

      for i in range(n_res):
          # 2D DWT decomposition
          cA, (cH, cV, cD) = pywt.dwt2(cA, wv_filt)

          # Estimate noise sigma
          sigma_n = np.median(np.abs(cD)) / 0.6745
          ns.append(sigma_n)

          # Estimate signal sigma
          sigma_s = np.std(cA)
          ss.append(sigma_s)

      return ns, ss

    def compare_noise(img, img2, n_res=1, wv_filt='db2'):

      ns_original, _ = nois_estimate_sigma_dwt(img, n_res, wv_filt)
      ns_enhanced, _ = nois_estimate_sigma_dwt(img2, n_res, wv_filt)

      print("🔍 Noise comparison per level:")

      for i in range(n_res):
          print(f"Level {i+1}:")
          print(f"Original noise sigma: {ns_original[i]:.4f}")
          print(f"Enhanced noise sigma: {ns_enhanced[i]:.4f}")
          diff = ns_enhanced[i] - ns_original[i]
          print(f"Pojacanje suma: {diff:.4f}\n")

      return ns_original, ns_enhanced, diff

    ns_original, ns_enhanced, diff = compare_noise(original, enhanced, n_res=4)

    #printing part
    print("🔍 **CONTRAST, SHARPNESS AND NOISE METRICS**")

    print("\n🔍 **METRICS OF CONTRAST, SHARPNESS AND NOISE**\n")
    print(f"{'Method':<20}{'Original':>12} → {'Enhanced':<12}")

    rms_orig = rms_contrast(original)
    rms_enh = rms_contrast(enhanced)
    print(f"{'RMS contrast':<20}{rms_orig:12.4f} → {rms_enh:<12.4f}")

    ten_orig = tenengrad(original)
    ten_enh = tenengrad(enhanced)
    print(f"{'Tenengrad':<20}{ten_orig:12.4f} → {ten_enh:<12.4f}")

    lap_orig = laplacian_variance(original)
    lap_enh = laplacian_variance(enhanced)
    print(f"{'Laplacian':<20}{lap_orig:12.4f} → {lap_enh:<12.4f}")

    mic_orig = michelson_contrast(original)
    mic_enh = michelson_contrast(enhanced)
    print(f"{'Michelson contrast':<20}{mic_orig:12.4f} → {mic_enh:<12.4f}")

    ent_orig = entropy(original)
    ent_enh = entropy(enhanced)
    print(f"{'Entropy':<20}{ent_orig:12.4f} → {ent_enh:<12.4f}")

    sigma_noise_orig = estimate_sigma_noise(original)
    sigma_noise_enh = estimate_sigma_noise(enhanced)
    print(
        f"{'Estimated noise level (lower increase is better)':<20}{sigma_noise_orig:12.4f} → {sigma_noise_enh:<12.4f}")

    ssim_val = ssim_index(original, enhanced)
    print(f"\nSSIM (structural similarity with the original): {ssim_val:.4f}")

    psnr_val = psnr(original.astype(np.uint16), enhanced.astype(np.uint16))
    print(f"\nPSNR (everything above 35dB is good): {psnr_val:.4f}")

    return {
        "RMS contrast (orig)": rms_orig,
        "RMS contrast (enh)": rms_enh,
        "Tenengrad (orig)": ten_orig,
        "Tenengrad (enh)": ten_enh,
        "Laplacian (orig)": lap_orig,
        "Laplacian (enh)": lap_enh,
        "Michelson (orig)": mic_orig,
        "Michelson (enh)": mic_enh,
        "Entropy (orig)": ent_orig,
        "Entropy (enh)": ent_enh,
        "Noise (orig)": sigma_noise_orig,
        "Noise (enh)": sigma_noise_enh,
        "SSIM": ssim_val,
        "PSNR": psnr_val,
        "Original noise sigma": ns_original,
        "Enhanced noise sigma": ns_enhanced,
        "Increase in noise sigma": diff
    }

