import os
import bz2
from typing import List, Tuple
from multiprocessing import Pool
from glob import glob
from PIL import Image
import tqdm
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector

LANDMARKS_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

ALIGNED = "aligned"
RAW = "raw"
BONAFIDE = "bonafide"


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
    for fname, ofname in tqdm.tqdm(pairs, position=process_num):
        for _, face_landmarks in enumerate(
            landmarks_detector.get_landmarks(fname), start=1
        ):
            os.makedirs(os.path.split(ofname)[0], exist_ok=True)
            image_align(fname, ofname, face_landmarks)


def getpairs(rdir: str, odir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    files = glob(os.path.join(rdir, RAW, "*", "*.png")) + glob(
        os.path.join(rdir, RAW, "*", "*.jpg")
    )

    for file in files:
        temp = file.split(RAW + "/")[1].replace("jpg", "png")

        if os.path.isfile(os.path.join(rdir, temp.replace("png", "JPG"))):
            Image.open(os.path.join(rdir, temp.replace("png", "JPG"))).convert(
                "RGB"
            ).save(os.path.join(rdir, temp))
            os.remove(os.path.join(rdir, temp.replace("png", "JPG")))
            continue

        ofname = os.path.join(odir, temp)
        os.makedirs(os.path.split(ofname)[0], exist_ok=True)
        if os.path.isfile(ofname):
            continue

        pairs.append((file, ofname))
    return pairs


def driver(
    CLEAN_DIR: str, printers: List[str] = ["digital"], num_process: int = 2
) -> None:
    args: List[Tuple[str, str]] = []
    for printer in printers:
        #         if os.path.isdir(os.path.join(CLEAN_DIR, printer, ALIGNED, RAW)):
        args.extend(
            getpairs(
                os.path.join(CLEAN_DIR, printer, BONAFIDE),
                os.path.join(CLEAN_DIR, printer, ALIGNED),
            )
        )

    if not args:
        print("No raw images to align:", CLEAN_DIR)
        return

    step = len(args) // num_process

    chunks = [args[x : x + step] for x in range(0, len(args), step)]
    with Pool(num_process) as p:
        p.map(align_images, enumerate(chunks))


if __name__ == "__main__":
    #     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/"
    #     driver(CLEAN_DIR)
    #     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/"
    #     driver(CLEAN_DIR)
    #     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/abc_database/"
    #     driver(CLEAN_DIR)
    #     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/narayan/digital/"
    #     subds = os.listdir(CLEAN_DIR)
    #     subds = [
    #         d for d in subds if "." not in d and os.path.isdir(os.path.join(CLEAN_DIR, d))
    #     ]
    #     driver(CLEAN_DIR, printers=subds)

    CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/lfc"
    driver(CLEAN_DIR)
#     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/ms40/"
#     driver(CLEAN_DIR)
#     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frill"
#     driver(CLEAN_DIR)
