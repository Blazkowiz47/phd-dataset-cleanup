import os
import json
import shutil
from os.path import join as pjoin
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import re


BONAFIDE_RAW_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/synonot/ONOT/digital"
MORPH_RAW_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/synonot/MONOT/Morphed/"
CLEAN_DIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/synonot/digital"


def clean_bonafide() -> List[str]:
    subjects: List[str] = []
    regex_for_subject_id = r"(.*)-G(.)-A(..)"
    dir = pjoin(CLEAN_DIR, "bonafide/raw")
    os.makedirs(dir, exist_ok=True)
    info_dict: Dict[str, Dict[str, str]] = {}
    for subjectid in tqdm(os.listdir(BONAFIDE_RAW_DIR)):
        for i, file in enumerate(Path(pjoin(BONAFIDE_RAW_DIR, subjectid)).rglob("*")):
            fname = file.name
            match = re.search(regex_for_subject_id, fname)
            if match:
                subjectid1 = match.group(1)
                gender1 = match.group(2)
                age1 = match.group(3)

                if subjectid1 not in info_dict:
                    info_dict[subjectid1] = {"gender": gender1, "age": age1}

            # shutil.copy(file, pjoin(dir, f"{subjectid}_{i}{file.suffix}"))
        subjects.append(subjectid)

    with open(pjoin(CLEAN_DIR, "gender_mapping.json"), "w+") as fp:
        json.dump(info_dict, fp)

    print(len(info_dict), len(subjects))
    return subjects


def clean_morphs() -> Dict[str, Dict[str, str]]:
    # filename = "M_S9176807-GM-A21-EEA-T03-S1-LH-C1-I0024-F00_S124298498-GM-A91-EEA-T14-S1-LH-C1-I0034-F00_C01_B30_W30_PA01_PM00_F00"
    regex_for_subject_id = r"M_(.*)-G(.)-A(..)-.*F00_(.*)-G(.)-A(..)-(.*)"
    # match = re.search(regex_for_subject_id, filename)
    info_dict: Dict[str, Dict[str, str]] = {}
    dir = pjoin(CLEAN_DIR, "morph/lmaubo/raw")
    os.makedirs(dir, exist_ok=True)
    done_imgs = set()
    for file in tqdm(Path(MORPH_RAW_DIR).rglob("*")):
        fname = file.name
        match = re.search(regex_for_subject_id, fname)
        if match:
            subjectid1 = match.group(1)
            gender1 = match.group(2)
            age1 = match.group(3)
            subjectid2 = match.group(4)
            gender2 = match.group(5)
            age2 = match.group(6)
            remaining = match.group(7)

            if subjectid1 not in info_dict:
                info_dict[subjectid1] = {"gender": gender1, "age": age1}

            if subjectid2 not in info_dict:
                info_dict[subjectid2] = {"gender": gender2, "age": age2}
            done_imgs.add(f"{subjectid1}_random-vs-{subjectid2}_random_{remaining}")
            shutil.move(
                file,
                pjoin(
                    dir,
                    f"{subjectid1}_random-vs-{subjectid2}_random_{remaining}",
                ),
            )
        else:
            print(f"Skipping {fname}")

    # with open(pjoin(CLEAN_DIR, "gender_mapping.json"), "w+") as fp:
    #     json.dump(info_dict, fp)
    print(len(done_imgs))
    return info_dict


if __name__ == "__main__":
    subjects = clean_bonafide()
    morphed_subjects = clean_morphs()
    # for subject in subjects:
    #     if subject not in morphed_subjects:
    #         print(subject)
    print(len(subjects), len(morphed_subjects))
