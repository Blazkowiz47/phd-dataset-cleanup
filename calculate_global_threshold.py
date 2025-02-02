"""
Calculate global FAR thresholds for arcface, adaface, and magface models.
Databases used: FRGC Complete, FERET, ABC
"""

import os
from typing import Dict, Iterable, List, Tuple
from multiprocessing import Process, Manager

import torch
from numpy.typing import NDArray
from tqdm import tqdm
import numpy as np
from torch.nn import CosineSimilarity
import matplotlib.pyplot as plt

RDIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets"
COSINE_SIM = CosineSimilarity(dim=1)


def load_data(backbone: str) -> Dict[str, List[Tuple[str, torch.Tensor]]]:
    subject_mappings: Dict[str, List[Tuple[str, torch.Tensor]]] = {}

    for i, sdir in enumerate(
        [
            f"{RDIR}/frgc_2/digital/bonafide/raw/frs/{backbone}",
            f"{RDIR}/feret/frs/{backbone}/digital/bonafide/test/",
            f"{RDIR}/feret/frs/{backbone}/digital/bonafide/train/",
        ]
    ):
        for subject in tqdm(os.listdir(sdir), desc=f"Processing {sdir}"):
            sid, _ = subject.split("_")

            if i:
                sid = f"FERET_{sid}"
            else:
                sid = f"FRGC_{sid}"

            if sid not in subject_mappings:
                subject_mappings[sid] = []
            subject_mappings[sid].append(
                (
                    os.path.join(sdir, subject),
                    torch.tensor(
                        np.load(os.path.join(sdir, subject)).squeeze().reshape((1, -1))
                    ),
                )
            )

    return subject_mappings


def get_genuine_pairs(
    subject_mappings: Dict[str, List[Tuple[str, torch.Tensor]]],
) -> List[float]:
    scores: List[float] = []
    for _, files in tqdm(subject_mappings.items()):
        for img1, emb1 in files:
            for img2, emb2 in files:
                if img1 != img2:
                    dist = COSINE_SIM(emb1, emb2).tolist()[0]
                    scores.append(dist)
    return scores


def get_imposter_pairs(
    subject_mappings: Dict[str, List[Tuple[str, torch.Tensor]]],
) -> List[float]:
    scores: List[float] = []
    for sid1, files1 in tqdm(subject_mappings.items()):
        for sid2, files2 in subject_mappings.items():
            if sid1 != sid2:
                for _, emb1 in files1:
                    for _, emb2 in files2:
                        dist = COSINE_SIM(emb1, emb2).tolist()[0]
                        scores.append(dist)
    return scores


def get_fars_and_frrs(
    genuine_scores: NDArray, imposter_scores: NDArray
) -> Tuple[NDArray, NDArray, NDArray]:
    thresholds = np.linspace(0, 1, 100001)
    fars = np.linspace(0, 1, 100001)
    frrs = np.linspace(0, 1, 100001)

    for i, threshold in tqdm(np.ndenumerate(thresholds), total=100001):
        far = np.sum(imposter_scores >= threshold) / len(imposter_scores)
        frr = np.sum(genuine_scores < threshold) / len(genuine_scores)
        fars[i] = far
        frrs[i] = frr

    return fars, frrs, thresholds


def plot_histograms(
    fars: NDArray, frrs: NDArray, thresholds: List[float], backbone: str
) -> None:
    plt.title(f"{backbone} global scores")
    plt.hist(fars, bins=100001, alpha=0.5, label="FAR", color="red")
    plt.hist(frrs, bins=100001, alpha=0.5, label="FRR", color="blue")
    for threshold in thresholds:
        plt.axvline(
            x=threshold,
            color="black",
            linestyle="--",
            label=f"Threshold {threshold:.4f}",
        )

    plt.savefig(
        f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/scores/{backbone}_gen_imp.png"
    )
    plt.close()


def driver() -> None:
    os.makedirs("scores", exist_ok=True)
    backbones = [
        # "magface",
        "adaface",
        # "arcface",
    ]

    for backbone in backbones:
        genuine_scores_path: str = f"scores/global_{backbone}_gen_scores.npy"
        imposter_scores_path: str = f"scores/global_{backbone}_imp_scores.npy"
        # data = load_data(backbone)
        genuine_scores = np.load(genuine_scores_path)
        imposter_scores = np.load(imposter_scores_path)
        fars, frrs, thresholds = get_fars_and_frrs(genuine_scores, imposter_scores)
        actual_threshold = [
            thresholds[np.argmin(np.abs(fars - 1e-3))],
            thresholds[np.argmin(np.abs(fars - 1e-4))],
            thresholds[np.argmin(np.abs(fars - 1e-5))],
        ]
        with open(
            f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/scores/global_{backbone}_threshods.txt",
            "w+",
        ) as f:
            f.write(f"Thresholds for {backbone} are :")
            f.write(f"1e-3: {actual_threshold[0]}")
            f.write(f"1e-4: {actual_threshold[1]}")
            f.write(f"1e-5: {actual_threshold[2]}")

        np.save(
            f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/scores/global_{backbone}_actual_thresholds.npy",
            np.array(actual_threshold),
        )

        plot_histograms(fars, frrs, actual_threshold, backbone)

        np.save(
            f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/scores/global_{backbone}_fars.npy",
            fars,
        )
        np.save(
            f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/scores/global_{backbone}_frrs.npy",
            frrs,
        )
        np.save(
            f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/scores/global_{backbone}_thresholds.npy",
            thresholds,
        )

        # if not os.path.isfile(genuine_scores_path):
        #     genuine_scores = get_genuine_pairs(data)
        #     np.save(genuine_scores_path, genuine_scores)
        #     del genuine_scores
        #
        # if not os.path.isfile(imposter_scores_path):
        #     imposter_scores = get_imposter_pairs(data)
        #     np.save(imposter_scores_path, imposter_scores)
        #     del imposter_scores


if __name__ == "__main__":
    driver()
