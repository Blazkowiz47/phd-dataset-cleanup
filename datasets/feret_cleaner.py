import json
import os
import shutil
from glob import glob
from typing import List

from tqdm import tqdm

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


def create_indices(fname: str, dirname: str, oname: str) -> None:
    with open(fname, "r") as fp:
        subjects = json.load(fp)

    files = glob(os.path.join(dirname, "*.jpg")) + glob(os.path.join(dirname, "*.png"))

    males = {k: v for k, v in subjects.items() if v["gender"] == "m"}
    females = {k: v for k, v in subjects.items() if v["gender"] == "f"}
    print(fname, len(males), len(females))
    for sid, _ in males.items():
        males[sid]["file"] = [
            os.path.split(f)[1]
            for f in files
            if os.path.split(f)[1].split("_")[0] == sid
        ]
    for sid, _ in females.items():
        females[sid]["file"] = [
            os.path.split(f)[1]
            for f in files
            if os.path.split(f)[1].split("_")[0] == sid
        ]

    pairs: List[str] = []

    for id1 in males:
        for id2 in males:
            if id1 == id2:
                continue
            pairs.append(f"{males[id1]['file'][0]},{males[id2]['file'][0]}\n")

    for id1 in females:
        for id2 in females:
            if id1 == id2:
                continue
            pairs.append(f"{females[id1]['file'][0]},{females[id2]['file'][0]}\n")

    with open(oname, "w+") as fp:
        fp.writelines(pairs)


def test() -> None:
    for ssplit in ["test", "train"]:
        subject_ids_in_train = {}
        for file in os.listdir(os.path.join(CLEAN_DIR, "digital", "bonafide", ssplit)):
            id, nm = file.split(".")[0].split("_")
            if id in subject_ids_in_train:
                subject_ids_in_train[id].append(nm)
            else:
                subject_ids_in_train[id] = [nm]
        print(ssplit, len(subject_ids_in_train.keys()))
        with open(os.path.join(CLEAN_DIR, f"{ssplit}_gender.json"), "r") as fp:
            data = json.load(fp)

        print(ssplit, len(data.keys()))
        print()


if __name__ == "__main__":
    #     clean_digital()
    #     get_bonafide_stats(
    #         os.path.join(CLEAN_DIR, "digital", BONAFIDE, RAW),
    #         os.path.join(CLEAN_DIR, "digital", "bonafide_stats.txt"),
    #     )
    #     clean_digital_morph()
    # create_indices(
    #     os.path.join(CLEAN_DIR, "test_gender.json"),
    #     os.path.join(CLEAN_DIR, "digital", ALIGNED, "test"),
    #     os.path.join(CLEAN_DIR, "test_index.csv"),
    # )
    # create_indices(
    #     os.path.join(CLEAN_DIR, "train_gender.json"),
    #     os.path.join(CLEAN_DIR, "digital", ALIGNED, "train"),
    #     os.path.join(CLEAN_DIR, "train_index.csv"),
    # )
    test()
