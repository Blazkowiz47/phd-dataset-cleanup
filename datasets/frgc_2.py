import bz2
import shutil
import json
import os
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm  # noqa: F811

from facenet_pytorch import MTCNN
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from frs import get_frs_initializers

times = ["Fall2002", "Fall2003", "Spring2003", "Spring2004"]

RAW_ROOT_FOLDER = "/mnt/cluster/nbl-datasets/face/FRGC-Complete/FRGC-2.0-dist/nd1/"
RAW_FOLDER = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc_2/digital/bonafide/raw"

LANDMARKS_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

BACKBONES = ["arcface", "adaface", "magface"]


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, "wb") as fp:
        fp.write(data)
    return dst_path


def align_images(args: Tuple[int, List[Tuple[str, str]]]) -> None:
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    process_num, pairs = args

    landmarks_model_path = "./models/temp/shape_predictor_68_face_landmarks.dat"
    #         get_file(
    #             "shape_predictor_68_face_landmarks.dat.bz2",
    #             LANDMARKS_MODEL_URL,
    #             cache_subdir="temp",
    #         )
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for fname, ofname in tqdm(pairs, position=process_num):
        for _, face_landmarks in enumerate(
            landmarks_detector.get_landmarks(fname), start=1
        ):
            os.makedirs(os.path.split(ofname)[0], exist_ok=True)
            image_align(fname, ofname, face_landmarks)


def facedetect(args: Tuple[int, List[Tuple[str, str]]]) -> None:
    mtcnn = MTCNN(image_size=224, device="cuda")
    pos, pairs = args
    for arg in tqdm(pairs, position=pos):
        fname, oname = arg
        img = Image.open(fname).convert("RGB")
        _ = mtcnn(img, save_path=oname)


def copy_raw_files_to_folder(
    src_folder: str, face_detect_folder: str, align_folder: str, frs_folder: str
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    face_detect_args: List[Tuple[str, str]] = []
    align_args: List[Tuple[str, str]] = []
    frs_args: List[Tuple[str, str]] = []
    for root, dirs, files in os.walk(src_folder):
        for dir in dirs:
            args1, args2, args3 = copy_raw_files_to_folder(
                os.path.join(root, dir), face_detect_folder, align_folder, frs_folder
            )
            face_detect_args.extend(args1)
            align_args.extend(args2)
            frs_args.extend(args3)

        for file in files:
            if (
                file.lower().endswith(".png")
                or file.lower().endswith(".jpg")
                or file.lower().endswith(".jpeg")
            ):
                src_file = os.path.join(root, file)
                face_detect_file = (
                    os.path.join(face_detect_folder, file.replace("d", "_"))
                    .replace("png", "jpg")
                    .replace("jpeg", "jpg")
                )
                align_file = (
                    os.path.join(align_folder, file.replace("d", "_"))
                    .replace("jpg", "png")
                    .replace("jpeg", "png")
                )
                frs_file = (
                    os.path.join(frs_folder, file.replace("d", "_"))
                    .replace("jpg", "npy")
                    .replace("JPG", "npy")
                    .replace("jpeg", "npy")
                    .replace("JPEG", "npy")
                    .replace("png", "npy")
                    .replace("PNG", "npy")
                )

                face_detect_args.append((src_file, face_detect_file))
                align_args.append((src_file, align_file))
                frs_args.append((face_detect_file, frs_file))

    return face_detect_args, align_args, frs_args


def frsextract(args: Tuple[int, Tuple[str, List[Tuple[str, str]]]]) -> None:
    pos, (backbone, pairs) = args
    get_model, get_features = get_frs_initializers(backbone)
    model = get_model()
    for arg in tqdm(pairs, position=pos):
        fname, oname = arg
        oname = oname.replace("/frs/", f"/frs/{backbone}/")
        if os.path.isfile(oname):
            continue

        try:
            feature = get_features(fname, model)
            os.makedirs(os.path.split(oname)[0], exist_ok=True)
            np.save(oname, feature)
        except Exception as e:
            print(e)


def frsdriver(args: List[Tuple[str, str]], num_process: int) -> None:
    step = len(args) // num_process
    chunks = [args[x : x + step] for x in range(0, len(args), step)]
    chunks = [(backbone, chunk) for backbone in BACKBONES for chunk in chunks]
    print(num_process)
    with Pool(num_process) as p:
        p.map(frsextract, enumerate(chunks))


def main(num_process: int = 8):
    all_face_detect_args: List[Tuple[str, str]] = []
    all_align_args: List[Tuple[str, str]] = []
    all_frs_args: List[Tuple[str, str]] = []
    face_detect_folder = os.path.join(RAW_FOLDER, "face_detected")
    align_folder = os.path.join(RAW_FOLDER, "aligned")
    frs_folder = os.path.join(RAW_FOLDER, "frs")

    for time in tqdm(times):
        src_folder = os.path.join(RAW_ROOT_FOLDER, time)
        os.makedirs(face_detect_folder, exist_ok=True)
        os.makedirs(align_folder, exist_ok=True)
        face_detect_args, align_args, frs_args = copy_raw_files_to_folder(
            src_folder, face_detect_folder, align_folder, frs_folder
        )
        all_face_detect_args.extend(face_detect_args)
        all_align_args.extend(align_args)
        all_frs_args.extend(frs_args)

    # frsdriver(all_frs_args, num_process)
    # return

    for args, callback in zip([all_align_args], [align_images]):
        step = len(args) // num_process
        chunks = [args[x : x + step] for x in range(0, len(args), step)]
        with Pool(num_process) as p:
            p.map(callback, enumerate(chunks))


def sortout_with_subjectids_and_gender():
    image_folder = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc_2/digital/bonafide/raw/sorted"
    src_folder = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc_2/digital/bonafide/raw/aligned"
    gender_mapping_file = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc_2/digital/bonafide/raw/aligned/gender.json"
    with open(gender_mapping_file, "r") as fp:
        gender_mapping = json.load(fp)

    for file in tqdm(os.listdir(src_folder)):
        sid = file.split("_")[0]
        gender = gender_mapping[sid]["gender"]

        dst_folder = os.path.join(image_folder, sid + "_" + gender)
        os.makedirs(dst_folder, exist_ok=True)
        dst_file = os.path.join(dst_folder, os.path.basename(file))
        src_file = os.path.join(src_folder, file)
        shutil.copy(src_file, dst_file)


if __name__ == "__main__":
    # main()
    sortout_with_subjectids_and_gender()
