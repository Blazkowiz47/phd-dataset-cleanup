import os
import shutil
from glob import glob
from tqdm import tqdm
from PIL import Image

ROOT_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/ms40/raw/"
CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/ms40/"
ALIGNED = "aligned"
BONAFIDE = "bonafide"
MORPH = "morph"
RAW = "raw"


def clean_digital_bon() -> None:
    sids = os.listdir()


if __name__ == "__main__":
    clean_digital_morph()
    clean_digital_bon()
