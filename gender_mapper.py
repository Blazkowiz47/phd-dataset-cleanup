import json
import os
from pathlib import Path
from typing import Dict, List

import getch


def get_mappings(
    subjects: Dict[str, Dict[str, List[str] | str]],
    fpath: str,
) -> Dict[str, Dict[str, List[str] | str]]:
    subjectid = 0
    sids = list(subjects.keys())
    sample = 0
    gender = None

    while subjectid < len(sids):
        sid = sids[subjectid]
        if not gender and "gender" in subjects[sid]:
            subjectid += 1
            continue

        fname = subjects[sid]["file"][sample]
        print(fname)
        os.system(f"wezterm imgcat {fname}")
        print("Is it male or female? type m/f:")
        gender = getch.getche()
        if gender == "^C":
            exit(0)
        if gender not in ["f", "m", "n"]:
            subjectid -= 1
            print("Going back")
            continue
        if gender == "n":
            sample = (sample + 1) % len(subjects[sid]["file"])
            continue

        print(f"assigning: {gender} to: {sid}")
        subjects[sid]["gender"] = gender
        subjectid += 1
        sample = 0
        gender = None

    with open(fpath, "w+") as fp:
        json.dump(subjects, fp)

    return subjects


def splitted_datasets(rdir: str, oname: str) -> None:
    if not os.path.isdir(rdir):
        return
    files = [
        str(file)
        for file in Path(rdir).rglob("*")
        if file.suffix.lower() == ".png" or file.suffix.lower() == ".jpg"
    ]
    subjects: Dict[str, Dict[str, List[str] | str]] = {}
    for file in files:
        sid = os.path.split(file)[1].split("_")[0]
        if sid not in subjects:
            subjects[sid] = {"file": [file]}
        else:
            subjects[sid]["file"].append(file)

    get_mappings(subjects, oname)


def unsplitted_datasets(rdir: str, oname: str) -> None:
    subjects: Dict[str, Dict[str, List[str] | str]] = {
        sid: {} for sid in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, sid))
    }
    for sid in subjects:
        sdir = os.path.join(rdir, sid)
        files = [
            str(file)
            for file in Path(sdir).rglob("*")
            if file.suffix.lower() == ".png" or file.suffix.lower() == ".jpg"
        ]
        subjects[sid] = {"file": files}

    get_mappings(subjects, oname)


if __name__ == "__main__":
    datasets = [
        #         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/abc_database/digital/aligned/",
        #         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/digital/aligned/",
        #         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frill/digital/aligned/",
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/aligned/",
    ]
    for dataset in datasets:
        for ssplit in ["train", "test"]:
            splitted_datasets(
                os.path.join(dataset, ssplit),
                os.path.join(
                    dataset.replace("digital/aligned/", ""), ssplit + "_gender.json"
                ),
            )

#     datasets = [
#         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/ms40/raw",
#     ]
#     for dataset in datasets:
#         unsplitted_datasets(
#             dataset,
#             os.path.join(dataset, "gender.json"),
#         )
