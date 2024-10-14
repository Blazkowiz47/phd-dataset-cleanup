import pandas as pd

files = ["./testindex.txt", "./trainindex.txt"]

for fname in files:
    with open(fname, "r") as fp:
        lines = fp.readlines()
        data = {"img1": [], "img2": []}
        for line in lines:
            s1, s2 = line.split(" ")
            s1 = './' + '/'.join(s1.split("\\")) 
            s2 = './' + '/'.join(s2.replace("\n", '').split("\\")) 
            data["img1"].append(s1)
            data["img2"].append(s2)

        df = pd.DataFrame(data)
        df.to_csv(fname.replace("txt", "csv"), header=False, index=False)
