import os
import shutil
from glob import glob

from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET

RAW_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/chokepoint/raw/"
CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/chokepoint/"
ALIGNED = "aligned"
BONAFIDE = "bonafide"
MORPH = "morph"
RAW = "raw"


def organise() -> None:
    subdirs = os.listdir(RAW_DIR)
    for subdir in subdirs:
        if not os.path.isdir(os.path.join(RAW_DIR, subdir)) or "ground" in subdir:
            continue
        ground_truth = os.path.join(RAW_DIR, "groundtruth", subdir + ".xml")
        pl, s, c = subdir.split("_")

        portal = pl[:2]
        status = pl[2]
        sequence = s[1]
        if sequence == "5":
            continue
        camera = c[1]

        tree = ET.parse(ground_truth)
        root = tree.getroot()

        print(root.tag)
        print(root.attrib)
        for frame in root:
            for person in frame.findall("person"):
                frame_name = frame.attrib["number"] + ".jpg"
                subid = person.attrib["id"]


if __name__ == "__main__":
    organise()
