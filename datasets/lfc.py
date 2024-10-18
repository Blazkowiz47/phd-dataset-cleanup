import os
import json
import shutil
from glob import glob
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
        test_sids = random.choices(sids, k=int(len(sids) * 0.3))
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
            file = os.path.join(train_dir, sid + "_1.bmp")
            if os.path.isfile(file):
                train_genders[sid] = {"file": file, "gender": gender[0]}
            else:
                test_genders[sid] = {"file": file, "gender": gender[0]}

    with open(os.path.join(CLEAN_DIR, "digital", "test_index.json"), "w+") as fp:
        json.dump(test_genders, fp)

    with open(os.path.join(CLEAN_DIR, "digital", "train_index.json"), "w+") as fp:
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


if __name__ == "__main__":
    clean_digital_bon()
    # get_gender()
    convert_to_jpg()
