from typing import Union

import cv2
import numpy as np

from utils.file import get_image


def draw_bbox(image: Union[str, np.ndarray], bbox: Union[np.ndarray, tuple]):
    if type(image) == str:
        image = get_image(image)

    color = (0, 0, 255)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    return image
