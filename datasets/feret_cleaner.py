import os
import shutil
from glob import glob
from tqdm import tqdm
from PIL import Image

ROOT_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/feret/"
CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret"
ALIGNED = "aligned"
BONAFIDE = "bonafide"
MORPH = "morph"
RAW = "raw"


def change_name(file) -> str:
    head, tail = os.path.split(file)
    tail = tail.removeprefix("scanned_").replace("-", "_")
    return os.path.join(head, tail)


def get_bonafide_stats(bdir: str, ofile: str) -> None:
    stats: str = ""
    ids = {}
    for ssplit in ["test", "train"]:
        images = glob(os.path.join(bdir, ssplit, "*.jpg"))
        if len(images) == 0:
            images = glob(os.path.join(bdir, ssplit, "*.png"))

        stats += f"{ssplit.capitalize()} split \n"
        stats += f"Total Images: {len(images)}\n"
        for image in images:
            id, nm = os.path.split(image)[1].split("_")
            if id in ids:
                ids[id].append(nm)
            else:
                ids[id] = [nm]
        stats += f"\tTotal identities: {len(ids.keys())}\n"
        mean_images = sum([len(x) for x in ids.values()]) / len(ids.keys())
        stats += f"\tAverage images per identity: {round(mean_images, 3)}\n\n"

    with open(ofile, "w+") as fp:
        fp.write(stats)

    print(stats)


def clean_digital() -> None:
    root_dir = os.path.join(ROOT_DIR, "digital")
    clean_dir = os.path.join(CLEAN_DIR, "digital")
    os.makedirs(os.path.join(clean_dir, BONAFIDE), exist_ok=True)
    os.makedirs(os.path.join(clean_dir, MORPH), exist_ok=True)
    os.makedirs(os.path.join(clean_dir, ALIGNED), exist_ok=True)
    os.makedirs(os.path.join(clean_dir, RAW), exist_ok=True)

    for fpath in tqdm(glob(os.path.join(root_dir, ALIGNED, "*", "*.png"))):
        opath = fpath.split(ALIGNED + "/")[1]
        opath = os.path.join(clean_dir, ALIGNED, opath)
        os.makedirs(os.path.split(opath)[0], exist_ok=True)
        shutil.copy(fpath, opath)

    for fpath in tqdm(glob(os.path.join(root_dir, BONAFIDE, "*", "*.jpg"))):
        opath = fpath.split(BONAFIDE + "/")[1]
        opath = os.path.join(clean_dir, RAW, opath)
        os.makedirs(os.path.split(opath)[0], exist_ok=True)
        shutil.copy(fpath, opath)


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


def clean_digital_raw(mdir: str, rdir: str):
    for ssplit in ["test", "train"]:
        dir = os.path.join(mdir, ssplit)
        osrdir = os.path.join(rdir, ssplit)
        os.makedirs(osrdir, exist_ok=True)
        for file in tqdm(glob(os.path.join(dir, "*.png"))):
            fname = clean_morph_name(os.path.split(file)[1])
            ofrname = os.path.join(osrdir, fname)
            shutil.copy(file, ofrname)


def clean_digital_morph() -> None:
    clean_digital_raw(
        os.path.join(ROOT_DIR, "digital", MORPH, "ubo"),
        os.path.join(CLEAN_DIR, "digital", MORPH, "lmaubo", RAW),
    )
    clean_digital_raw(
        os.path.join(ROOT_DIR, "digital", MORPH, "mipgan2"),
        os.path.join(CLEAN_DIR, "digital", MORPH, "mipgan2", RAW),
    )
    clean_digital_raw(
        os.path.join(ROOT_DIR, "digital", MORPH, "MorDiff"),
        os.path.join(CLEAN_DIR, "digital", MORPH, "mordiff", RAW),
    )
    clean_digital_raw(
        os.path.join(ROOT_DIR, "digital", MORPH, "greedy"),
        os.path.join(CLEAN_DIR, "digital", MORPH, "greedy", RAW),
    )


def check_differences(dir1, dir2):
    for ssplit in ["test", "train"]:
        dir1ids = {}
        images = glob(os.path.join(dir1, ssplit, "*.jpg"))
        if len(images) == 0:
            images = glob(os.path.join(dir1, ssplit, "*.png"))

        for image in images:
            id, nm = os.path.split(image)[1].split("_")
            if id in dir1ids:
                dir1ids[id].append(nm)
            else:
                dir1ids[id] = [nm]

        dir2ids = {}
        images = glob(os.path.join(dir2, ssplit, "*.jpg"))
        if len(images) == 0:
            images = glob(os.path.join(dir2, ssplit, "*.png"))
        for image in images:
            id, nm = os.path.split(image)[1].split("_")
            if id in dir2ids:
                dir2ids[id].append(nm)
            else:
                dir2ids[id] = [nm]

        print(ssplit, len(dir1ids.keys()), len(dir2ids.keys()))
        for id in dir2ids:
            if id not in dir1ids:
                print("Not present:", id)


if __name__ == "__main__":
    #     clean_digital()
    get_bonafide_stats(
        os.path.join(CLEAN_DIR, "digital", BONAFIDE, RAW),
        os.path.join(CLEAN_DIR, "digital", "bonafide_stats.txt"),
    )
    clean_digital_morph()
