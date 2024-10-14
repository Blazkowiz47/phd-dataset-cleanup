import os

from tqdm import tqdm


resultsdir = "./Results"


def execute(img1: str, img2: str) -> bool:
    print("Performing:", f"{img1} {img2}")
    with open("./IndexFile.txt", "w+") as fp:
        fp.writelines([f"{img1} {img2}"])

    os.system("MorphedImageGenerator.bat")

    found = False
    for img in os.listdir(resultsdir):
        if not img.lower().endswith(".png"):
            continue
        if "W0.50" not in img:
            continue

        img = img.removeprefix("M_").split(".")[0].removesuffix("_W0")
        s1, i1, s2, i2 = img.split("_")

        if (
            s1 + "_" + i1 == img1.split(".")[0].split("\\")[-1]
            and s2 + "_" + i2 == img2.split(".")[0].split("\\")[-1]
        ):
            found = True
            print("Found")
            break

    return found


def mainloop() -> None:
    with open("./tIndexFile.txt", "r") as fp:
        morphpairs = fp.readlines()

    for mpair in tqdm(morphpairs):
        img1, img2 = mpair.split(" ")
        success = False
        while not success:
            success = execute(img1, img2)


if __name__ == "__main__":
    os.chdir("D:/UBO_morpher/")
    mainloop()
