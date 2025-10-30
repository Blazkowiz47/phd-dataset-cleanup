from typing import List, Dict, Tuple
from multiprocessing import Pool
import os
import csv
from pathlib import Path
from random import choice
from tqdm import tqdm
from morph_images import get_morph_driver, perform_morphing
# from ffhq_align_images import align_images
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


def align_perturbed_images():
    num_process = 8
    # First generate the csv's
    align_image_pairs: List[Tuple[str, str]] = []
    for dataset_dir in RDIR.glob("*"):
        if not dataset_dir.is_dir():
            continue
        for experiment_dir in dataset_dir.glob("*"):
            if not experiment_dir.is_dir():
                continue
            for protocol in ["test", "train"]:
                protocol_dir = experiment_dir / protocol
                if not protocol_dir.is_dir():
                    continue
                print(protocol_dir)
                continue
                for image in protocol_dir.glob("*"):
                    if image.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                        continue
                    image_path = str(image)
                    o_path = image_path.replace("bonafide", "bona_aligned")
                    os.makedirs(os.path.split(o_path)[0], exist_ok=True)
                    align_image_pairs.append(
                        (image_path, image_path.replace("bonafide", "bona_aligned"))
                    )

    # step = len(align_image_pairs) // num_process
    #
    # chunks = [
    #     align_image_pairs[x : x + step] for x in range(0, len(align_image_pairs), step)
    # ]
    # with Pool(num_process) as p:
    #     p.map(align_images, enumerate(chunks))


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
        for subdir in ["bonafide"]:
            for ssplit in ["test", "train"]:
                idir = rdir / dataset / "digital" / subdir / ssplit
                odir = rdir / dataset / "digital" / subdir / "resized_112x112" / ssplit
                odir.mkdir(parents=True, exist_ok=True)
                perform_resize(idir, odir)


def face_detect_images():
    morphs = ["greedy", "pipe", "mordiff"]
    for morph in morphs:
        for dataset_dir in RDIR.glob("*"):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            if "frgc" not in str(dataset_dir).lower():
                continue
            for experiment in dataset_dir.glob("*"):
                if not (RDIR / experiment).is_dir():
                    continue
                if not str(experiment).endswith(
                    ("BPDA_EOT", "DCT_HF", "DWT_HF", "PGD")
                ):
                    continue

                if "original" == experiment.name:
                    continue

                lmaubo_morph_dir = dataset_dir / experiment / "aligned"
                for ssplit in ["train", "test"]:
                    protocol_dir = lmaubo_morph_dir / ssplit
                    if not protocol_dir.is_dir():
                        continue
                    for morph_type in ["morph", "morph_unaligned_bonafide"]:
                        src_dir = str(protocol_dir).replace(
                            "aligned", f"{morph_type}/{morph}"
                        )
                        out_dir = (
                            str(src_dir)
                            .replace("test", "facedetected/test")
                            .replace("train", "facedetected/train")
                        )
                        print("=" * 100)
                        print("Src dir:", src_dir)
                        print("Out dir:", out_dir)
                        print("=" * 100)
                        from mtcnn_face_detect import get_pairs, face_detect_wrapper

                        pairs = get_pairs(src_dir, out_dir)
                        with Pool(2) as p:
                            _ = list(
                                tqdm(
                                    p.imap(face_detect_wrapper, pairs), total=len(pairs)
                                )
                            )


if __name__ == "__main__":
    # first_morphing_greedy()
    # morph_perturbed_images()
    # resize_all()
    # align_perturbed_images()
    face_detect_images()
