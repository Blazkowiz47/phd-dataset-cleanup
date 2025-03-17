import os
from pathlib import Path
from glob import glob
from multiprocessing import Pool
from typing import List, Tuple

from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm
import numpy as np
from math import atan2, degrees, cos, sin

from yolo_pytorch.face_detector import YoloDetector

BONAFIDE = "bonafide"
MORPH = "morph"
RAW = "raw"


def getpairs(dir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    files = Path(dir).rglob("*")
    for file in files:
        if file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        file = str(file)
        temp = file.replace(dir, os.path.join(dir, "facedetect")).replace("png", "jpg")
        pairs.append((file, temp))
    return pairs


def align_and_crop_face(image, bboxes, landmarks, margin=0.2) -> NDArray:
    """
    Aligns the largest face upright using eye landmarks (90° or 180° rotation only) and crops it.

    :param image: The input image (BGR, numpy array).
    :param bboxes: List of bounding boxes [[x_min, y_min, x_max, y_max], ...].
    :param landmarks: List of landmark sets [[(x1, y1), (x2, y2), ...], ...].
    :param margin: Margin as a percentage of the bbox size.
    :return: Aligned and cropped face image.
    """
    # Select the largest bounding box
    largest_bbox = get_largest_bbox(bboxes)
    if largest_bbox is None:
        return None  # No faces detected

    # Get corresponding landmarks
    bbox_index = bboxes.index(largest_bbox)
    face_landmarks = landmarks[bbox_index]

    # Get bounding box coordinates
    x_min, y_min, x_max, y_max = largest_bbox
    w, h = x_max - x_min, y_max - y_min

    # Get image dimensions
    img_h, img_w = image.shape[:2]

    # Extract eye landmarks
    left_eye, right_eye = (
        face_landmarks[0],
        face_landmarks[1],
    )  # (x, y) for left & right eye

    # Determine rotation based on bbox shape & eye positioning
    if w > h:  # Face is horizontal, needs rotation
        if right_eye[1] > left_eye[1]:
            rotated_img = np.rot90(image, -1)  # 90° CW
            new_bbox = [y_min, img_w - x_max, y_max, img_w - x_min]  # Adjust bbox
        elif right_eye[1] < left_eye[1]:
            rotated_img = np.rot90(image, 1)  # 90° CCW
            new_bbox = [img_h - y_max, x_min, img_h - y_min, x_max]  # Adjust bbox
        else:
            rotated_img = np.rot90(image, 2)  # 180° rotation
            new_bbox = [img_w - x_max, img_h - y_max, img_w - x_min, img_h - y_min]
    else:  # Face is already upright
        rotated_img = image
        new_bbox = largest_bbox
    # Extract new bbox coordinates after rotation
    x_min, y_min, x_max, y_max = new_bbox

    # Expand bbox with margin
    margin_x = int(margin * (x_max - x_min))
    margin_y = int(margin * (y_max - y_min))

    x_min = max(x_min - margin_x, 0)
    y_min = max(y_min - margin_y, 0)
    x_max = min(x_max + margin_x, rotated_img.shape[1])
    y_max = min(y_max + margin_y, rotated_img.shape[0])

    # Crop the aligned face
    cropped_face = rotated_img[y_min:y_max, x_min:x_max]

    return cropped_face


def get_largest_bbox(bboxes):
    """
    Selects the largest bounding box based on area.

    :param bboxes: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
    :return: Bounding box with the maximum area.
    """
    max_area = 0
    largest_bbox = None

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        area = (x_max - x_min) * (y_max - y_min)
        if area > max_area:
            max_area = area
            largest_bbox = bbox

    return largest_bbox


def facedetect(args: Tuple[int, List[Tuple[str, str]]]) -> None:
    yolodetector = YoloDetector(target_size=720, device="cuda", min_face=90)
    pos, pairs = args
    for arg in tqdm(pairs, position=pos):
        fname, oname = arg
        image = np.array(Image.open(fname).convert("RGB"))
        # = eyes, nose, lips corners
        bboxes, landmarks = yolodetector.predict(image)
        aligned_face = align_and_crop_face(image, bboxes[0], landmarks[0], margin=0.2)
        os.makedirs(os.path.dirname(oname), exist_ok=True)
        if aligned_face is not None:
            Image.fromarray(aligned_face).save(oname)


def driver(num_process: int) -> None:
    args: List[Tuple[str, str]] = []
    rdir = "/home/ubuntu/datasets/test"
    for dataset in os.listdir(rdir):
        ddir = os.path.join(rdir, dataset)
        if os.path.isdir(ddir):
            args.extend(getpairs(ddir))

    print(len(args))
    step = len(args) // num_process
    chunks = [args[x : x + step] for x in range(0, len(args), step)]
    with Pool(num_process) as p:
        p.map(facedetect, enumerate(chunks))


if __name__ == "__main__":
    num_process = 6

    driver(num_process)
