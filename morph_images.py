import sys
import subprocess
import os
import csv
from typing import Callable, Tuple

from torch import e


def get_morph_driver(morph: str) -> Callable[[Tuple[int, str, str, str]], None]:
    if morph == "lma":
        from morphs.lma import driver

        sys.path.append("./morphs/lma/")

        return driver
    if morph == "ubo":
        from morphs.lmaubo import driver

        sys.path.append("./morphs/lmaubo/")

        return driver
    if morph == "ladimo":
        from morphs.ladimo import driver

        sys.path.append("./morphs/ladimo/")

        return driver
    if morph == "mipgan1":
        from morphs.mipgan1 import driver

        sys.path.append("./morphs/mipgan1/")

        return driver
    if morph == "mipgan2":
        from morphs.mipgan2 import driver

        sys.path.append("./morphs/mipgan2/")

        return driver
    if morph == "pipe":
        from morphs.pipe import driver

        sys.path.append("./morphs/pipe/")

        return driver
    if morph == "mordiff":
        from morphs.mordiff import driver

        sys.path.append("./morphs/mordiff/")

        return driver
    if morph == "greedy":
        from morphs.greedy import driver

        sys.path.append("./morphs/greedy/")

        return driver

    raise NotImplementedError(f"Morpher {morph} not implemented")


def perform_morphing(
    morph: str, src_dir: str, morph_list_csv: str, output_dir: str
) -> None:
    morpher = get_morph_driver(morph)
    flag = True
    while flag:
        try:
            morpher((0, src_dir, morph_list_csv, output_dir))
            flag = False
        except KeyboardInterrupt:
            break
        except ValueError:
            pass


def main() -> None:
    #     perform_morphing(
    #         "ladimo",
    #         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/aligned/test/",
    #         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/test_index.csv",
    #         "./test_morphs/ladimo",
    #     )

    ssplits = ["test", "train"]
    morphs = ["mipgan2", "mipgan1", "lma"]
    #     morphs = ["pipe", "mordiff"]
    datasets = ["feret", "frgc", "abc_database", "frill", "ms40"]
    #     datasets = ["frgc", "abc_database", "frill", "ms40"]
    top_ks = [6, 3, 4, 3, 3]
    for morph in morphs:
        for ssplit in ssplits:
            # Normal datasets
            for dataset, top_k in zip(datasets, top_ks):
                if os.path.isfile(
                    f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/{dataset}/{ssplit}_index.csv"
                ):
                    rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/{dataset}/digital/aligned/{ssplit}/"
                    csv_file = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/{dataset}/{ssplit}_index.csv"
                    odir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/{dataset}/digital/morph/{morph}/{ssplit}"
                    temp_csv_file = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/{dataset}/top_3_{ssplit}_index.csv"
                    #                     if not os.path.isfile(temp_csv_file):
                    subprocess.call(
                        f'source ~/miniconda3/etc/profile.d/conda.sh; conda activate torch;python get_topk_frs_match.py -o "{temp_csv_file}" -i "{csv_file}" -r "{rdir}" -k {top_k}',
                        shell=True,
                        executable="/bin/bash",
                    )
                    perform_morphing(morph, rdir, temp_csv_file, odir)

            # narayan dataset
            narayan_rdir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/narayan/digital"
            for dataset in os.listdir(narayan_rdir):
                if not os.path.isdir(os.path.join(narayan_rdir, dataset)):
                    continue
                aligned_dir = os.path.join(narayan_rdir, dataset, "aligned", ssplit)
                csvfile = os.path.join(narayan_rdir, dataset, f"{ssplit}_index.csv")
                output_dir = os.path.join(narayan_rdir, dataset, "morph", morph, ssplit)
                if os.path.isfile(csvfile):
                    perform_morphing(morph, aligned_dir, csvfile, output_dir)


#     perform_morphing(
#         "pipe",
#         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/aligned/test/",
#         "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/test_index.csv",
#         "./test_morphs/pipe",
#     )
if __name__ == "__main__":
    main()
