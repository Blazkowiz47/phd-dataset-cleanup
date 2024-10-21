import os
import json
import shutil
from glob import glob
from typing import List
from tqdm import tqdm
import numpy as np
from PIL import Image
import random

ROOT_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/lfc/raw/"
CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/lfc/"
ALIGNED = "aligned"
BONAFIDE = "bonafide"
MORPH = "morph"
RAW = "raw"


def clean_digital_bon() -> None:
    for gender in ["male", "female"]:
        dir = os.path.join(ROOT_DIR, gender)
        sids = [sid for sid in os.listdir(dir) if os.path.isdir(os.path.join(dir, sid))]
        test_sids = random.sample(sids, k=int(len(sids) * 0.3))
        train_sids = [sid for sid in sids if sid not in test_sids]
        for sids, ssplit in zip([train_sids, test_sids], ["train", "test"]):
            odir = os.path.join(CLEAN_DIR, "digital", "bonafide", ssplit)
            for sid in tqdm(sids):
                idir = os.path.join(dir, sid)
                for imgname in glob(os.path.join(idir, "*.bmp")) + glob(
                    os.path.join(idir, "*.BMP")
                ):
                    imgname = os.path.split(imgname)[1]
                    imgpath = os.path.join(idir, imgname)
                    os.makedirs(odir, exist_ok=True)
                    opath = os.path.join(odir, imgname.replace("ID_", ""))
                    shutil.copy(imgpath, opath)


def get_gender() -> None:
    test_genders = {}
    train_genders = {}
    train_dir = os.path.join(CLEAN_DIR, "digital", "bonafide", "train")
    test_dir = os.path.join(CLEAN_DIR, "digital", "bonafide", "test")

    for gender in ["male", "female"]:
        dir = os.path.join(ROOT_DIR, gender)
        sids = [
            sid.replace("Subject_", "")
            for sid in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, sid))
        ]

        for sid in sids:
            files = glob(os.path.join(train_dir, sid + "_*.bmp"))
            if files:
                train_genders[sid] = {"file": files, "gender": gender[0]}
            else:
                files = glob(os.path.join(test_dir, sid + "_*.bmp"))
                test_genders[sid] = {"file": files, "gender": gender[0]}

    with open(os.path.join(CLEAN_DIR, "test_gender.json"), "w+") as fp:
        json.dump(test_genders, fp)

    with open(os.path.join(CLEAN_DIR, "train_gender.json"), "w+") as fp:
        json.dump(train_genders, fp)


def convert_to_jpg() -> None:
    rdir = os.path.join(CLEAN_DIR, "digital", "bonafide")
    for ssplit in ["test", "train"]:
        dir = os.path.join(rdir, ssplit, "*.bmp")
        for img in tqdm(glob(dir)):
            image = np.array(Image.open(img))
            image = image.squeeze()
            image = np.stack([image, image, image], axis=2)
            Image.fromarray(image).save(img.replace("bmp", "jpg"))
            os.remove(img)


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
                f"{males[id1]['file'][0].replace('bmp', 'jpg')},{males[id2]['file'][0].replace('bmp', 'jpg')}\n"
            )

    for id1 in females:
        for id2 in females:
            if id1 == id2:
                continue
            pairs.append(
                f"{females[id1]['file'][0].replace('bmp', 'jpg')},{females[id2]['file'][0].replace('bmp', 'jpg')}\n"
            )

    with open(oname, "w+") as fp:
        fp.writelines(pairs)


if __name__ == "__main__":
    random.seed(2024)
    clean_digital_bon()
    get_gender()
    convert_to_jpg()
    create_indices(
        os.path.join(CLEAN_DIR, "test_gender.json"),
        os.path.join(CLEAN_DIR, "digital", BONAFIDE, "test"),
        os.path.join(CLEAN_DIR, "test_index.csv"),
    )
    create_indices(
        os.path.join(CLEAN_DIR, "train_gender.json"),
        os.path.join(CLEAN_DIR, "digital", BONAFIDE, "train"),
        os.path.join(CLEAN_DIR, "train_index.csv"),
    )
