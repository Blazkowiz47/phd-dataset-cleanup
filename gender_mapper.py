import json
import os
import getch
from pathlib import Path
from typing import Dict


def get_mappings(subjects: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    i = 0
    sids = list(subjects.keys())

    while i < len(sids):
        sid = sids[i]
        fname = subjects[sid]["file"]
        print(fname)
        os.system(f"wezterm imgcat {fname}")
        print("Is it male or female? type m/f:")
        gender = getch.getche()
        if gender == "^C":
            exit(0)
        if gender not in ["f", "m"]:
            i -= 1
            print("Going back")
            continue

        print(f"assigning: {gender} to: {sid}")
        subjects[sid]["gender"] = gender
        i += 1

    return subjects


def splitted_datasets(rdir: str, oname: str) -> None:
    files = [
        str(file)
        for file in Path(rdir).rglob("*")
        if file.suffix == ".png" or file.suffix == "jpg"
    ]
    subjects: Dict[str, Dict[str, str]] = {}
    for file in files:
        sid = os.path.split(file)[1].split("_")[0]
        if sid not in subjects:
            subjects[sid] = {"file": file}

    get_mappings(subjects)
    with open(oname, "w+") as fp:
        json.dump(subjects, fp)


def unsplitted_datasets(rdir: str, oname: str) -> None:
    subjects: Dict[str, Dict[str, str]] = {
        sid: {} for sid in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, sid))
    }
    for sid in subjects:
        sdir = os.path.join(rdir, sid)
        files = [
            str(file)
            for file in Path(sdir).rglob("*")
            if file.suffix.lower() == ".png" or file.suffix.lower() == "jpg"
        ]
        subjects[sid] = {"file": files[0]}

    get_mappings(subjects)
    with open(oname, "w+") as fp:
        json.dump(subjects, fp)


if __name__ == "__main__":
    datasets = [
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/abc_database/digital/aligned/",
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/digital/aligned/",
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

    datasets = [
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/ms40/raw",
    ]
    for dataset in datasets:
        unsplitted_datasets(
            dataset.replace("raw", ""),
            os.path.join(dataset, "gender.json"),
        )
