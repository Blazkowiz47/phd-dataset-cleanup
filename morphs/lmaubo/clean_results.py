import os

resultsdir = "./Results"


def execute() -> None:
    for img in os.listdir(resultsdir):
        if not img.lower().endswith(".png"):
            continue
        if img.endswith("W0.50_B0.50_AR_CE.png"):
            continue

        os.remove(os.path.join(resultsdir, img))


if __name__ == "__main__":
    os.chdir("D:/UBO_morpher/")
    execute()
