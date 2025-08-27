import cv2


image_data = cv2.imread("/Users/macbookair/Code/Adaptive_CLAHE_MSE/Adaptive_CLAHE_MSE/data/raw/CHEST_PA_N_v2.tiff", cv2.IMREAD_UNCHANGED)
image = image_data  # already cropped
print(image_data.max())