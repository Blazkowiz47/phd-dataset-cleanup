import os
import json
import shutil
from glob import glob
from typing import List
from tqdm import tqdm
from PIL import Image
import random

ROOT_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/ms40/raw/"
CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/ms40"
ALIGNED = "aligned"
BONAFIDE = "bonafide"
MORPH = "morph"
RAW = "raw"


def clean_digital_bon() -> None:
    with open(os.path.join(ROOT_DIR, "gender.json"), "r") as fp:
        subjects = json.load(fp)
    males = {k: v for k, v in subjects.items() if v["gender"] == "m"}
    females = {k: v for k, v in subjects.items() if v["gender"] == "f"}

    maleids = list(males.keys())
    femaleids = list(females.keys())

    train_males = random.sample(maleids, k=25)
    train_females = random.sample(femaleids, k=3)

    test_males = [sid for sid in maleids if sid not in train_males]
    test_females = [sid for sid in femaleids if sid not in train_females]

    train_genders = {
        **{k: subjects[k] for k in train_males},
        **{k: subjects[k] for k in train_females},
    }
    test_genders = {
        **{k: subjects[k] for k in test_males},
        **{k: subjects[k] for k in test_females},
    }

    with open(os.path.join(CLEAN_DIR, "test_gender.json"), "w+") as fp:
        json.dump(test_genders, fp)
    with open(os.path.join(CLEAN_DIR, "train_gender.json"), "w+") as fp:
        json.dump(train_genders, fp)

    for subjects, ssplit in zip([test_genders, train_genders], ["test", "train"]):
        for sid, info in tqdm(subjects.items()):
            for file in info["file"]:
                head = os.path.join(CLEAN_DIR, "digital", BONAFIDE, RAW, ssplit)
                tail = os.path.split(file)[1]
                try:
                    tail = str(sid) + "_" + tail.split("_")[1]
                except IndexError:
                    tail = str(sid) + "_" + tail
                tail = tail.lower()
                ofname = os.path.join(head, tail)
                os.makedirs(head, exist_ok=True)
                shutil.copy(file, ofname)


def get_bonafide_stats(bdir: str, ofile: str) -> None:
    stats: str = ""
    for ssplit in ["test", "train"]:
        images = glob(os.path.join(bdir, ssplit, "*.jpg"))
        if len(images) == 0:
            images = glob(os.path.join(bdir, ssplit, "*.png"))

        stats += f"{ssplit.capitalize()} split \n"
        stats += f"Total Images: {len(images)}\n"
        ids = {}
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


def create_indices(fname: str, dirname: str, oname: str) -> None:
    with open(fname, "r") as fp:
        subjects = json.load(fp)

    files = glob(os.path.join(dirname, "*.jpg")) + glob(os.path.join(dirname, "*.png"))

    males = {k: v for k, v in subjects.items() if v["gender"] == "m"}
    females = {k: v for k, v in subjects.items() if v["gender"] == "f"}
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
            pairs.append(
                f"{males[id1]['file'][-1].replace('jpg', 'png')},{males[id2]['file'][-1].replace('jpg', 'png')}\n"
            )

    for id1 in females:
        for id2 in females:
            if id1 == id2:
                continue
            pairs.append(
                f"{females[id1]['file'][-1].replace('jpg', 'png')},{females[id2]['file'][-1].replace('jpg', 'png')}\n"
            )

    with open(oname, "w+") as fp:
        fp.writelines(pairs)


if __name__ == "__main__":
    #     random.seed(2024)
    #     clean_digital_bon()
    #     get_bonafide_stats(
    #         os.path.join(CLEAN_DIR, "digital", BONAFIDE, RAW),
    #         os.path.join(CLEAN_DIR, "digital", "bonafide_stats.txt"),
    #     )
    create_indices(
        os.path.join(CLEAN_DIR, "test_gender.json"),
        os.path.join(CLEAN_DIR, "digital", BONAFIDE, RAW, "test"),
        os.path.join(CLEAN_DIR, "test_index.csv"),
    )
    create_indices(
        os.path.join(CLEAN_DIR, "train_gender.json"),
        os.path.join(CLEAN_DIR, "digital", BONAFIDE, RAW, "train"),
        os.path.join(CLEAN_DIR, "train_index.csv"),
    )
