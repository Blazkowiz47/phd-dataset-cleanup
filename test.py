import os
from glob import glob
from tqdm import tqdm
import shutil


RDIR = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/"

DATASETS = ["feret", "frgc"]
SSPLITS = ["test", "train"]
MORPHS = ["greedy", "pipe", "mordiff", "mipgan1", "mipgan2", "lma"]


def clean() -> None:
    for ssplit in SSPLITS:
        for dataset in DATASETS:
            for printer in ["digital"]:
                odir = os.path.join(RDIR, dataset, printer, "bonafide", "raw", ssplit)
                os.makedirs(odir, exist_ok=True)
                for file in tqdm(
                    glob(
                        os.path.join(
                            RDIR, dataset, printer, "bonafide", ssplit, "*.png"
                        )
                    ),
                    desc=f"Bonafide for {dataset}",
                ):
                    shutil.move(file, os.path.join(odir, os.path.split(file)[1]))

                with open(
                    os.path.join(RDIR, dataset, f"top_3_{ssplit}_index.csv")
                ) as fp:
                    pair_csv_data = fp.readlines()

                for morph in MORPHS:
                    odir = os.path.join(
                        RDIR, dataset, printer, "morph", morph, "raw", ssplit
                    )
                    os.makedirs(odir, exist_ok=True)
                    rdir = os.path.join(RDIR, dataset, printer, "morph", morph, ssplit)

                    #                     morph_pairs = []
                    #                     for morph_pair in pair_csv_data:
                    #                         img1, img2 = morph_pair.split(",")
                    #                         fname = (
                    #                             img1.split(".")[0] + "-vs-" + img2.split(".")[0] + ".png"
                    #                         )
                    #                         morph_pairs.append(fname)
                    #                     count = 0
                    #
                    #                     for fname in glob(os.path.join(odir, "*.png")):
                    #                         if os.path.split(fname)[1] not in morph_pairs:
                    #                             os.remove(fname)
                    #                     continue

                    for morph_pair in tqdm(
                        pair_csv_data, desc=f"Morph for {dataset} {morph}"
                    ):
                        img1, img2 = morph_pair.split(",")
                        fname = (
                            img1.split(".")[0] + "-vs-" + img2.split(".")[0] + ".png"
                        )

                        if os.path.isfile(os.path.join(rdir, fname)):
                            shutil.move(
                                os.path.join(rdir, fname),
                                os.path.join(odir, os.path.join(odir, fname)),
                            )


if __name__ == "__main__":
    clean()
