import cv2
import numpy as np

# Blue mask boundaries on BGR color profile.
BLUE_MASK_BOUNDARIES = np.array(([175, 115, 50], [255, 255, 255]))
# Green mask boundaries on BGR color profile.
GREEN_MASK_BOUNDARIES = np.array(([15, 5, 25], [25, 255, 95]), dtype="uint8")
# Brightness/blue mask boundaries on HSV color profile.
BRIGHTNESS_MASK_BOUNDARIES = np.array(([90, 5, 185], [130, 50, 255]), dtype="uint8")


def image_masker(image):
    """
    Function that extract red, green and brightness channel masks from an image.
    :param np.ndarray image: Image for mask extraction.
    :return: tuple of np.ndarray : blue, green and brightness masks from image.
    """
    blue_channel_image_mask = cv2.inRange(image, BLUE_MASK_BOUNDARIES[0], BLUE_MASK_BOUNDARIES[1])
    green_channel_image_mask = cv2.inRange(image, GREEN_MASK_BOUNDARIES[0], GREEN_MASK_BOUNDARIES[1])
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_channel_image_mask = cv2.inRange(hsv_image, BRIGHTNESS_MASK_BOUNDARIES[0], BRIGHTNESS_MASK_BOUNDARIES[1])
    return blue_channel_image_mask, green_channel_image_mask, brightness_channel_image_mask


def image_vertical_cropper(image, keeping_percentage):
    """
    Function that crops an image percentage vertically (up to bottom).
    :param image: Image to be cropped.
    :param keeping_percentage: Image percentage to be kept up to bottom.
    :return: np.ndarray : Cropped image.
    """
    # Target image dimension extraction.
    height, width = image.shape
    # Calculates the number of rows to be kept according to given percentage.
    boundary_row_index = int(height * keeping_percentage)
    # Crops and return image.
    return image[:boundary_row_index]
