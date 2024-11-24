import argparse
import os
from typing import Any, Dict, Tuple, List

import numpy as np
from tqdm import tqdm

from frs import get_frs_initializers

parser = argparse.ArgumentParser()
parser.add_argument(
    "-r",
    "--rdir",
    type=str,
    #     default="/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/digital/aligned/test/",
)
parser.add_argument(
    "-i",
    "--input_csv",
    #     default="/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/test_index.csv",
    type=str,
)
parser.add_argument(
    "-o",
    "--output_csv",
    #     default="./temp_morph.csv",
    type=str,
)
parser.add_argument(
    "-b",
    "--backbone",
    default="arcface",
    required=False,
    type=str,
)
parser.add_argument(
    "-k",
    "--top_k",
    default=5,
    type=int,
)


def driver(
    rdir: str, input_csv_file: str, output_csv_file: str, backbone: str, top_k: int
):
    with open(input_csv_file, "r") as f:
        pairs = f.readlines()

    morph_mappings: Dict[str, List[str]] = {}
    for pair in pairs:
        if "," not in pair:
            continue
        img1, img2 = pair.strip().split(",")
        if img1 not in morph_mappings:
            morph_mappings[img1] = []

        morph_mappings[img1].append(img2)

    get_model, get_features = get_frs_initializers(backbone)
    model = get_model()

    frs_feature_mappings: Dict[str, Any] = {}
    for fname in tqdm(list(morph_mappings.keys())):
        frs_feature_mappings[fname] = get_features(os.path.join(rdir, fname), model)

    top_matching_mappings: Dict[str, List[str]] = {}
    for id1 in frs_feature_mappings.keys():
        scores = []
        for id2 in morph_mappings[id1]:
            f1, f2 = frs_feature_mappings[id1], frs_feature_mappings[id2]
            f1, f2 = f1 / np.linalg.norm(f1), f2 / np.linalg.norm(f2)
            cos_score = (f1 * f2).sum()
            scores.append(cos_score)

        sorted_indexes = np.argsort(scores)
        top_k_indexes = sorted_indexes[-top_k:]
        top_matching_mappings[id1] = []
        for index in top_k_indexes:
            top_matching_mappings[id1].append(morph_mappings[id1][index])

    total_morphs: int = 0
    with open(output_csv_file, "w+") as fp:
        for id1 in top_matching_mappings:
            for id2 in top_matching_mappings[id1]:
                fp.write(f"{id1},{id2}\n")
                total_morphs += 1
    print(f"Total morphs: {total_morphs}")


if __name__ == "__main__":
    args = parser.parse_args()
    driver(args.rdir, args.input_csv, args.output_csv, args.backbone, args.top_k)
