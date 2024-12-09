import os
from glob import glob
from multiprocessing import Pool
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

from frs import get_frs_initializers


BONAFIDE = "bonafide"
MORPH = "morph"
BACKBONES = [
    "adaface",
    "magface",
    "arcface",
]
FACEDETECT = "facedetect"


def getpairs(dir: str, odir: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    files = glob(os.path.join(dir, "*.png")) + glob(os.path.join(dir, "*.jpg"))
    for file in files:
        temp = os.path.split(file)[1].replace("jpg", "npy").replace("png", "npy")
        temp = os.path.join(odir, temp)
        pairs.append((file, temp))

    return pairs


def frsextract(args: Tuple[int, Tuple[str, List[Tuple[str, str]]]]) -> None:
    pos, (backbone, pairs) = args
    get_model, get_features = get_frs_initializers(backbone)
    model = get_model()
    for arg in tqdm(pairs, position=pos):
        fname, oname = arg
        feature = get_features(fname, model)
        os.makedirs(os.path.split(oname)[0], exist_ok=True)
        np.save(oname, feature)


def driver(CLEAN_DIR: str, printers: List[str], num_process: int) -> None:
    args: Dict[str, List[Tuple[str, str]]] = {}

    for backbone in BACKBONES:
        args[backbone] = []

    odir = os.path.join(CLEAN_DIR, "frs")
    for backbone in BACKBONES:
        for printer in printers:
            for ssplit in ["test", "train"]:
                if os.path.isdir(
                    os.path.join(CLEAN_DIR, printer, "bonafide", FACEDETECT, ssplit)
                ):
                    args[backbone].extend(
                        getpairs(
                            os.path.join(
                                CLEAN_DIR, printer, "bonafide", FACEDETECT, ssplit
                            ),
                            os.path.join(
                                odir, backbone, printer, "bonafide", FACEDETECT, ssplit
                            ),
                        )
                    )
                if os.path.isdir(os.path.join(CLEAN_DIR, printer, "bonafide", ssplit)):
                    args[backbone].extend(
                        getpairs(
                            os.path.join(CLEAN_DIR, printer, "bonafide", ssplit),
                            os.path.join(odir, backbone, printer, "bonafide", ssplit),
                        )
                    )

                if os.path.isdir(os.path.join(CLEAN_DIR, printer, "morph")):
                    for morph in os.listdir(os.path.join(CLEAN_DIR, printer, "morph")):
                        if os.path.isdir(
                            os.path.join(CLEAN_DIR, printer, "morph", morph, FACEDETECT)
                        ):
                            args[backbone].extend(
                                getpairs(
                                    os.path.join(
                                        CLEAN_DIR,
                                        printer,
                                        "morph",
                                        morph,
                                        FACEDETECT,
                                        ssplit,
                                    ),
                                    os.path.join(
                                        odir,
                                        backbone,
                                        printer,
                                        "morph",
                                        morph,
                                        FACEDETECT,
                                        ssplit,
                                    ),
                                )
                            )
                        if os.path.isdir(
                            os.path.join(CLEAN_DIR, printer, "morph", morph)
                        ):
                            args[backbone].extend(
                                getpairs(
                                    os.path.join(
                                        CLEAN_DIR, printer, "morph", morph, ssplit
                                    ),
                                    os.path.join(
                                        odir, backbone, printer, "morph", morph, ssplit
                                    ),
                                )
                            )

    for backbone, arg in args.items():
        step = len(arg) // num_process
        chunks = [arg[x : x + step] for x in range(0, len(arg), step)]
        chunks = [(backbone, chunk) for chunk in chunks]
        print(num_process)
        with Pool(num_process) as p:
            p.map(frsextract, enumerate(chunks))


if __name__ == "__main__":
    num_process = 6

    printers = ["dnp", "rico"]
    CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/"
    driver(CLEAN_DIR, printers, num_process)

    #     printers = ["dnp", "digital", "rico"]
    CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/"
    driver(CLEAN_DIR, printers, num_process)

    CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/narayan/"
    printers = ["digital"]
    for printer in printers:
        dir = os.path.join(CLEAN_DIR, printer)
        subds = os.listdir(dir)
        subds = [
            d for d in subds if "." not in d and os.path.isdir(os.path.join(dir, d))
        ]
        driver(dir, subds, num_process)
