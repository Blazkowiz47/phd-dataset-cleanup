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


def driver() -> None:
    os.makedirs("scores", exist_ok=True)
    backbones = [
        "arcface",
        "magface",
        "adaface",
    ]

    for backbone in backbones:
        genuine_scores_path: str = f"scores/global_{backbone}_gen_scores.npy"
        imposter_scores_path: str = f"scores/global_{backbone}_imp_scores.npy"
        data = load_data(backbone)

        if not os.path.isfile(genuine_scores_path):
            genuine_scores = get_genuine_pairs(data)
            np.save(genuine_scores_path, genuine_scores)
            del genuine_scores

        if not os.path.isfile(imposter_scores_path):
            imposter_scores = get_imposter_pairs(data)
            np.save(imposter_scores_path, imposter_scores)
            del imposter_scores


if __name__ == "__main__":
    driver()
