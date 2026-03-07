import cv2
import numpy as np

def calculate_brightness(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate average brightness
    brightness = np.mean(gray)

    return brightness

