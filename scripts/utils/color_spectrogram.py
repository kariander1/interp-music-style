
from PIL import Image
import cv2
import numpy as np

# Load the grayscale spectrogram image
image_path =  "images/style/accordion/accordion.png"
output_path = "images/style/accordion/accordion_color.png"


# Load the grayscale image
img = Image.open(image_path).convert("L")  # Convert to grayscale
img_array = np.array(img)

# Normalize to 0–255 and convert to uint8 if needed
if img_array.dtype != np.uint8:
    img_array = 255 * (img_array - img_array.min()) / (img_array.max() - img_array.min())
    img_array = img_array.astype(np.uint8)

# Apply OpenCV colormap (e.g., INFERNO)
colored_img = cv2.applyColorMap(img_array, cv2.COLORMAP_INFERNO)

# Save the result
cv2.imwrite(output_path, colored_img)
