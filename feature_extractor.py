import cv2
import numpy as np
from preprocessor import image_masker, image_vertical_cropper
from util import get_image_area


def feature_extractor(image_mask):
    """
    Function that extracts average feature presence from an image mask.
    :param ndarray image_mask: Channel mask from an image (0 and 255 values only).
    :return: ndarray : Feature average presence on image.
    """
    # Image normalization.
    normalized_image_mask = image_mask / 255
    # Calculates image area (width x height)
    image_area = get_image_area(image_mask)
    # Calculates and return the average feature presente on image.
    return np.sum(normalized_image_mask / image_area)


def feature_vector_generator(image):
    """
    Function that generate feature vector for specific use case (basic nature/city landscape recognition model).
    :param ndarray image: Target image to extract its feature vector.
    :return: list of np.ndarray : Target image feature vector.
    """
    # Image mask generation from green channel, blue channel and brightness channel.
    blue_image_mask, green_image_mask, brightness_image_mask = image_masker(image)
    # Compound mask that combines blue mask and brightness to model sky recognition.
    compound_image_mask = cv2.bitwise_and(blue_image_mask, blue_image_mask, mask=brightness_image_mask)
    # Image portion for sky detection.
    sky_area_mask = image_vertical_cropper(compound_image_mask, 0.30)
    # Calculates each feature and return image corresponding vector.
    return [
        feature_extractor(sky_area_mask),
        feature_extractor(green_image_mask),
        feature_extractor(blue_image_mask)
    ]
