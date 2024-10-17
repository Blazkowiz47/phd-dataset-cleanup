import json
import sys
import os
from pathlib import Path
from typing import Dict


def get_mappings(subjects: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    for sid in subjects:
        fname = subjects[sid]["file"]
        print(fname)
        os.system(f"wezterm imgcat {fname}")
        gender = input("Is it male or female? type m/f:")
        subjects[sid]["gender"] = gender

    return subjects


def driver(rdir: str, oname: str) -> None:
    sys.path.append(rdir)
    files = [str(file) for file in Path(rdir).rglob("*") if file.suffix == ".png"]
    subjects: Dict[str, Dict[str, str]] = {}
    for file in files:
        sid = file.split("_")[0]
        if sid not in subjects:
            subjects[sid] = {"file": file}

    get_mappings(subjects)
    with open(oname, "w+") as fp:
        json.dump(subjects, fp)


if __name__ == "__main__":
    for ssplit in ["train", "test"]:
        driver(
            f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/digital/aligned/{ssplit}/",
            f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/{ssplit}_gender.json",
        )
