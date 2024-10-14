import os

command = "python encode_images_MIPGAN.py --morph_list_CSV=test.csv --src_dir=./train/ --generated_images_dir=./morph_train/"

with open("./train_index.csv", "r") as fp:
    lines = fp.readlines()

for i in range(0, len(lines), 16):
    endid = min(i + 15, len(lines))
    x = lines[i : endid + 1]
    with open("./test.csv", "w+") as fp:
        fp.writelines(x)
    os.system(command)
