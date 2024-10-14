import os
import shutil
from glob import glob
from tqdm import tqdm
from PIL import Image

ROOT_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/PostProcess_Data/"
CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/postprocessdata"
ALIGNED = "aligned"
BONAFIDE = "bonafide"
MORPH = "morph"
RAW = "raw"


def clean_morph_name(name: str) -> str:
    # scanned
    if "scanned_" in name:
        name = name.removeprefix("scanned_")

    # pipe
    if "MDFP" in name:
        name = name.removeprefix("MDFP_")

    # mordiff
    if "MMDF" in name:
        name = name.removeprefix("MMDF_")

    # lmaubo
    if "M" in name:
        name = name.removeprefix("M_")
        temp = name.split(".")
        name = temp[0].removesuffix("_W0") + "." + temp[-1]

    # lma
    if "d" in name:
        if "_" in name:
            name = name.replace("_", "-vs-")
        name = name.replace("d", "_")
    return name


def clean_digital_raw(dir: str, osrdir: str):
    os.makedirs(osrdir, exist_ok=True)
    for file in tqdm(glob(os.path.join(dir, "*.png"))):
        fname = clean_morph_name(os.path.split(file)[1])
        ofrname = os.path.join(osrdir, fname)
        ofrname = os.path.join(osrdir, fname.replace("png", "jpg"))
        Image.open(file).save(ofrname)


def clean_digital_lma(mdir: str, osrdir: str):
    dir = os.path.join(mdir, "Face")
    os.makedirs(osrdir, exist_ok=True)
    for file in tqdm(glob(os.path.join(dir, "*.png"))):
        fname = os.path.split(file)[1]
        ofrname = os.path.join(osrdir, fname.replace("png", "jpg"))
        Image.open(file).save(ofrname)


def clean_digital_morph() -> None:
    for ssplit in ["Test", "Train"]:
        clean_digital_raw(
            os.path.join(ROOT_DIR, "cleaned_data", "digital", ssplit, MORPH, "After"),
            os.path.join(CLEAN_DIR, "digital_estimate", MORPH, "after", ssplit.lower()),
        )
        clean_digital_raw(
            os.path.join(ROOT_DIR, "cleaned_data", "digital", ssplit, MORPH, "Before"),
            os.path.join(
                CLEAN_DIR, "digital_estimate", MORPH, "before", ssplit.lower()
            ),
        )

    for ssplit in ["Test", "Train"]:
        clean_digital_lma(
            os.path.join(ROOT_DIR, "Digital", "After", "Mor", ssplit),
            os.path.join(CLEAN_DIR, "digital", MORPH, "after", ssplit.lower()),
        )
        clean_digital_lma(
            os.path.join(ROOT_DIR, "Digital", "Before", "Mor", ssplit),
            os.path.join(CLEAN_DIR, "digital", MORPH, "before", ssplit.lower()),
        )


def clean_digital_bon() -> None:
    for ssplit in ["Test", "Train"]:
        clean_digital_lma(
            os.path.join(ROOT_DIR, "Digital", "After", "Bon", ssplit),
            os.path.join(CLEAN_DIR, "digital", BONAFIDE, "after", ssplit.lower()),
        )
        clean_digital_lma(
            os.path.join(ROOT_DIR, "Digital", "Before", "Bon", ssplit),
            os.path.join(CLEAN_DIR, "digital", BONAFIDE, "before", ssplit.lower()),
        )


if __name__ == "__main__":
    clean_digital_morph()
    clean_digital_bon()
