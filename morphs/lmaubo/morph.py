import os
import shutil
from typing import Tuple

from tqdm import tqdm


resultsdir = "./morphs/lmaubo/Results"


def driver(args: Tuple[int, str, str, str]):
    process_num, src_dir, morph_list_csv, output_dir = args
    with open(morph_list_csv, "r") as fp:
        morph_list = fp.readlines()

    for pair in tqdm(morph_list, position=process_num):
        if not pair.strip():
            continue

        splited_pair = pair.strip().split(",")
        img1_path = splited_pair[0]
        img2_path = splited_pair[1]
        img1 = os.path.join(src_dir, img1_path)
        img2 = os.path.join(src_dir, img2_path)
        temp = (
            os.path.split(img1)[1].split(".")[0]
            + "-vs-"
            + os.path.split(img2)[1].split(".")[0]
        )
        output = os.path.join(output_dir, temp + ".png")
        success = False
        while not success:
            success = execute(img1, img2, output)


def execute(img1: str, img2: str, output: str) -> bool:
    with open("./morphs/lmaubo/IndexFile.txt", "w+") as fp:
        fp.writelines([f"{img1} {img2}"])

    os.makedirs(resultsdir, exist_ok=True)
    os.system(
        f"./morphs/lmaubo/MorphedImageGenerator.exe ./morphs/lmaubo/IndexFile.txt {resultsdir}"
    )

    found = False
    for img in os.listdir(resultsdir):
        file = img
        if not img.lower().endswith(".png"):
            continue
        if "W0.50" not in img:
            continue

        img = img.removeprefix("M_").split(".")[0].removesuffix("_W0")
        s1, i1, s2, i2 = img.split("_")

        if (
            s1 + "_" + i1 == os.path.split(img1)[1].split(".")[0]
            and s2 + "_" + i2 == os.path.split(img2)[1].split(".")[0]
        ):
            found = True
            print("Found")
            break

        os.rename(
            file,
            os.path.join(
                resultsdir,
                os.path.split(img1)[1]
                + "-vs-"
                + os.path.split(img2)[1]
                + "."
                + file.split(".")[-1],
            ),
        )
        shutil.copy(file, output)

    return found
