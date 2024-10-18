import os
import shutil
from glob import glob

from PIL import Image
from tqdm import tqdm

ROOT_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/narayan/raw/"
CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/narayan/"
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
    images = glob(
        os.path.join(ROOT_DIR, "*", "*", "*", "*", "Morph", "*.JPG"), recursive=True
    )
    rdir = os.path.join(CLEAN_DIR, "digital")

    for fname in tqdm(images):
        rel_path = fname.removeprefix(ROOT_DIR)
        ssplit, gender, subsec, subid, _, file = rel_path.split("/")
        dir = os.path.join(rdir, subsec, ALIGNED, RAW, ssplit)
        os.makedirs(dir, exist_ok=True)
        shutil.copy(fname, os.path.join(dir, subid.split("_")[1] + "_0.jpg"))

    images = glob(
        os.path.join(ROOT_DIR, "*", "*", "*", "*", "Probe", "*.JPG"), recursive=True
    )
    rdir = os.path.join(CLEAN_DIR, "digital")

    for fname in tqdm(images):
        rel_path = fname.removeprefix(ROOT_DIR)
        ssplit, gender, subsec, subid, _, file = rel_path.split("/")
        dir = os.path.join(rdir, subsec, BONAFIDE, RAW, ssplit)
        os.makedirs(dir, exist_ok=True)
        shutil.copy(
            fname,
            os.path.join(dir, subid.split("_")[1] + file.removeprefix("DSC").lower()),
        )

    images = glob(
        os.path.join(ROOT_DIR, "*", "*", "*", "*", "*", "Morph", "*.JPG"),
        recursive=True,
    )
    rdir = os.path.join(CLEAN_DIR, "digital")

    for fname in tqdm(images):
        rel_path = fname.removeprefix(ROOT_DIR)
        ssplit, gender, subsec, subsubsec, subid, _, file = rel_path.split("/")
        dir = os.path.join(rdir, subsubsec, ALIGNED, RAW, ssplit)
        os.makedirs(dir, exist_ok=True)
        shutil.copy(fname, os.path.join(dir, subid.split("_")[1] + "_0.jpg"))

    images = glob(
        os.path.join(ROOT_DIR, "*", "*", "*", "*", "*", "Probe", "*.JPG"),
        recursive=True,
    )
    rdir = os.path.join(CLEAN_DIR, "digital")

    for fname in tqdm(images):
        rel_path = fname.removeprefix(ROOT_DIR)
        ssplit, gender, subsec, subsubsec, subid, _, file = rel_path.split("/")
        dir = os.path.join(rdir, subsubsec, BONAFIDE, RAW, ssplit)
        os.makedirs(dir, exist_ok=True)
        shutil.copy(
            fname,
            os.path.join(dir, subid.split("_")[1] + file.removeprefix("DSC").lower()),
        )


def reorder_morphinglist() -> None:
    rdir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/To_Morphing/Narayan_DB/Source_Dataset/morphing_list"
    files = glob(
        os.path.join(
            rdir,
            "**",
            "*.csv",
        ),
        recursive=True,
    )

    for fname in files:
        print(fname.removeprefix(rdir + "/"))
        ssplit, gender, subsec, file = fname.removeprefix(rdir + "/").split("/")
        odir = os.path.join(CLEAN_DIR, "digital", subsec)
        #         os.remove(os.path.join(odir, gender.lower() + "_" + ssplit + "_" + file))

        with open(fname, "r") as fp:
            content = fp.readlines()

        ofile = os.path.join(
            odir, ("train" if "train" in ssplit.lower() else "test") + "_index.csv"
        )

        #         if os.path.exists(ofile):
        #             os.remove(ofile)
        #         continue

        with open(ofile, "a+") as fp:
            for line in content:
                fp.write(line.replace("Sub_", "").lower().replace(".jpg", "_0.png"))
            fp.write("\n")


if __name__ == "__main__":
    #     clean_digital_bon()
    reorder_morphinglist()
