import os
import pickle
from cv2 import imread
from typing import List

import numpy as np

IMAGE_TYPES = ("jpg", "jpeg", "png", "bmp", "tif", "tiff")


def get_file_list(base_path, filetypes=IMAGE_TYPES, as_path=False):
    files = os.listdir(base_path)
    files = [fn for fn in files if os.path.splitext(fn)[-1].lstrip('.').lower() in filetypes]
    return files if not as_path else [os.path.join(base_path, fn) for fn in files]


def read_text_file(fn):
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Path does not exist: {fn}")

    with open(fn, "r") as fp:
        lines = [line.strip() for line in fp]
    return lines


def write_text_file(fn, lines: List[str]):
    lines = [f"{l}\n" for l in lines if not l.endswith('\n')]
    with open(fn, 'w+') as f:
        f.writelines(lines)


def read_object_file(fn):
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Path does not exist: {fn}")

    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    return obj


def write_object_file(fn, object: dict):
    with open(fn, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_image(image_file, to_rgb=False):
    img = imread(image_file)
    if to_rgb:
        img = img[:, :, ::-1]
    return img

def tensor_to_cvimg(float_tensor):
    return np.multiply(float_tensor.permute(1, 2, 0).cpu().numpy(), 255).astype(np.uint8)
