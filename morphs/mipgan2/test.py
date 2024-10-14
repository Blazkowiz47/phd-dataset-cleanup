import pandas as pd

files = ["./testindex.txt", "./trainindex.txt"]

for fname in files:
    with open(fname, "r") as fp:
        lines = fp.readlines()
        data = {"img1": [], "img2": []}
        for line in lines:
            s1, s2 = line.split(" ")
            s1 = s1.split("\\")[-1].removeprefix("Images\\")
            s2 = s2.split("\\")[-1].removeprefix("Images\\").removesuffix("\n")
            data["img1"].append(s1)
            data["img2"].append(s2)

        df = pd.DataFrame(data)
        df.to_csv(fname.replace("txt", "csv"), header=False, index=False)
