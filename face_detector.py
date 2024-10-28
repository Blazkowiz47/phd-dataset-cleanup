import os
from glob import glob
from multiprocessing import Pool
from typing import List, Tuple

from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm

from facenet_pytorch import MTCNN

BONAFIDE = "bonafide"
MORPH = "morph"
RAW = "raw"


def getpairs(dir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    files = (
        glob(os.path.join(dir, RAW, "*", "*.png"))
        + glob(os.path.join(dir, RAW, "*", "*.jpg"))
        + glob(os.path.join(dir, RAW, "*", "*.JPG"))
    )
    for file in files:
        temp = file.split(RAW + "/")[1].replace("png", "jpg")
        pairs.append((file, os.path.join(dir, temp)))
    return pairs


def normalise(x: NDArray) -> NDArray:
    return (x - x.min()) / (x.max() - x.min())


def facedetect(args: Tuple[int, List[Tuple[str, str]]]) -> None:
    mtcnn = MTCNN(image_size=224, device="cuda")
    pos, pairs = args
    for arg in tqdm(pairs, position=pos):
        fname, oname = arg
        img = Image.open(fname).convert("RGB")
        _ = mtcnn(img, save_path=oname)


def driver(CLEAN_DIR: str, printers: List[str], num_process: int) -> None:
    args: List[Tuple[str, str]] = []
    for printer in printers:
        if os.path.isdir(os.path.join(CLEAN_DIR, printer, "bonafide", RAW)):
            args.extend(getpairs(os.path.join(CLEAN_DIR, printer, "bonafide")))

        if os.path.isdir(os.path.join(CLEAN_DIR, printer, "morph")):
            for morph in os.listdir(os.path.join(CLEAN_DIR, printer, "morph")):
                if os.path.isdir(os.path.join(CLEAN_DIR, printer, "morph", morph, RAW)):
                    args.extend(
                        getpairs(os.path.join(CLEAN_DIR, printer, "morph", morph))
                    )

    step = len(args) // num_process
    chunks = [args[x : x + step] for x in range(0, len(args), step)]
    with Pool(num_process) as p:
        p.map(facedetect, enumerate(chunks))


if __name__ == "__main__":
    num_process = 6

    #     printers = ["digital"]
    #     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/"
    #     driver(CLEAN_DIR, printers, num_process)
    #
    #     printers = ["dnp", "digital", "rico"]
    #     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/"
    #     driver(CLEAN_DIR, printers, num_process)

    printers = ["digital"]
    CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frill/"
    driver(CLEAN_DIR, printers, num_process)

#     CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/narayan/"
#     printers = ["digital"]
#     for printer in printers:
#         dir = os.path.join(CLEAN_DIR, printer)
#         subds = os.listdir(dir)
#         subds = [
#             d for d in subds if "." not in d and os.path.isdir(os.path.join(dir, d))
#         ]
#         driver(dir, subds, num_process)
