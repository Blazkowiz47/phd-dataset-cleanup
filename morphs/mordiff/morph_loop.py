import os
from typing import Tuple
from tqdm import tqdm
from .morph_two_images import morph_two_images


def driver(args: Tuple[int, str, str, str]):
    process_num, src_dir, morph_list_csv, output_dir = args
    with open(morph_list_csv, "r") as fp:
        morph_list = fp.readlines()
        morph_list = morph_list[:50]  # limit for testing

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
        os.makedirs(output_dir, exist_ok=True)
        output = os.path.join(output_dir, temp + ".png")
        if os.path.isfile(output):
            continue
        morph_two_images(img1, img2, output)
