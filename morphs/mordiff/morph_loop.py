import os
import sys
from typing import List, Tuple
from tqdm import tqdm


def morph_loop(morphs: List[Tuple[str, str, str]]) -> None:
    sys.path.append("./")
    for line in tqdm(morphs):
        img1, img2 = line.rstrip().split(" ")
        img1 = img1.replace("jpg", "png")
        img2 = img2.replace("jpg", "png")
        name1 = img1.split(".")[-2].split("/")[-1]
        name2 = img2.split(".")[-2].split("/")[-1]
        ofname = f"morphed_{name1}_and_{name2}.png"
        if os.path.isfile(f"../feret/morph/{ssplit}/{ofname}"):
            continue
        os.system(
            f"python morph_two_images.py --img1 {os.path.join('./aligned/', img1)} --img2 {os.path.join('./aligned/', img2)} --output ../feret/morph/{ssplit}"
        )
