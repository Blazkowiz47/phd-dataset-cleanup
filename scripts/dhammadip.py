from typing import List, Dict, Tuple
from multiprocessing import Pool
import os
import csv
from pathlib import Path
from random import choice
from tqdm import tqdm
from morph_images import get_morph_driver, perform_morphing
from ffhq_align_images import align_images
from PIL import Image

RDIR = Path("/mnt/cluster/nbl-datasets/face-morphing/Dhammadip/img-perturbed/")


def morph_perturbed_images():
    morphs = ["greedy", "pipe", "mordiff"]
    for morph in morphs:
        for dataset_dir in RDIR.glob("*"):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for experiment in dataset_dir.glob("*"):
                if not (RDIR / experiment).is_dir():
                    continue
                if "original" == experiment.name:
                    continue

                lmaubo_morph_dir = dataset_dir / experiment / "aligned"
                for ssplit in ["train", "test"]:
                    protocol_dir = lmaubo_morph_dir / ssplit
                    if not protocol_dir.is_dir():
                        continue

                    csv_dir = (
                        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/"
                        f"FaceMoprhingDatabases/cleaned_datasets/{dataset.lower()}"
                    )
                    csv_file_name = f"{ssplit}_index.csv"
                    csv_file = os.path.join(csv_dir, csv_file_name)
                    src_dir = str(protocol_dir)
                    out_dir = str(protocol_dir).replace("aligned", f"morph/{morph}")
                    print("=" * 100)
                    print("Csv file:", csv_file)
                    print("Src dir:", src_dir)
                    print("Out dir:", out_dir)
                    print("=" * 100)

                    continue
                    os.makedirs(out_dir, exist_ok=True)
                    perform_morphing(morph, src_dir, csv_file, out_dir)


def morph_perturbed_frgc_images():
    RDIR = RDIR / "FRGC"
    morphs = ["greedy", "pipe", "mordiff"]
    for morph in morphs:
        for experiment in RDIR.glob("*"):
            if not (RDIR / experiment).is_dir():
                continue

            lmaubo_morph_dir = RDIR / experiment / "bonafide/LMA"
            for protocol, csv_file_name in zip(
                ["ICAO_P1", "ICAO_P2"], ["test_index.csv", "train_index.csv"]
            ):
                protocol_dir = lmaubo_morph_dir / protocol
                if not protocol_dir.is_dir():
                    continue
                csv_dir = (
                    "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/"
                    "FaceMoprhingDatabases/cleaned_datasets/frgc"
                )
                csv_file = os.path.join(csv_dir, csv_file_name)
                src_dir = str(protocol_dir).replace("bonafide/LMA", "aligned")
                out_dir = str(protocol_dir).replace("bonafide/LMA", f"morph/{morph}")
                print("Csv file:", csv_file)
                print("Src dir:", src_dir)
                print("Out dir:", out_dir)
                os.makedirs(out_dir, exist_ok=True)
                perform_morphing(morph, src_dir, csv_file, out_dir)


def align_perturbed_frgc_images():
    num_process = 8
    # First generate the csv's
    align_image_pairs: List[Tuple[str, str]] = []
    for experiment in RDIR.glob("*"):
        if not (RDIR / experiment).is_dir():
            continue

        lmaubo_morph_dir = RDIR / experiment / "bonafide/LMA"
        for protocol in ["ICAO_P1", "ICAO_P2"]:
            protocol_dir = lmaubo_morph_dir / protocol
            if not protocol_dir.is_dir():
                continue
            for image in protocol_dir.glob("*"):
                if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue
                image_path = str(image)
                o_path = image_path.replace("bonafide/LMA", "aligned")
                os.makedirs(os.path.split(o_path)[0], exist_ok=True)
                align_image_pairs.append(
                    (image_path, image_path.replace("bonafide/LMA", "aligned"))
                )

    step = len(align_image_pairs) // num_process

    chunks = [
        align_image_pairs[x : x + step] for x in range(0, len(align_image_pairs), step)
    ]
    with Pool(num_process) as p:
        p.map(align_images, enumerate(chunks))


def perform_resize(idir: Path, odir: Path):
    for image in tqdm(idir.glob("*"), desc=str(idir)):
        if image.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        image = image.relative_to(idir)
        opath = odir / image
        Image.open(idir / image).resize((112, 112)).save(opath)


def resize_all():
    datasets = ["ms40"]
    rdir = Path(
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/"
    )
    for dataset in datasets:
        for subdir in [ "bonafide"]:
            for ssplit in ["test", "train"]:
                idir = rdir / dataset / "digital" / subdir / ssplit
                odir = rdir / dataset / "digital" / subdir / "resized_112x112" / ssplit
                odir.mkdir(parents=True, exist_ok=True)
                perform_resize(idir, odir)


if __name__ == "__main__":
    # first_morphing_greedy()
    # morph_perturbed_images()
    resize_all()
