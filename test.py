# from datasets.frgc_2 import main
#
# if __name__ == "__main__":
#     main()

import os
import numpy as np


for sdir in [
    "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/bonafide/raw/test/",
    "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/bonafide/raw/train/",
    "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc_2/digital/bonafide/raw/face_detected",
]:
    subject_ids = {}
    for file in os.listdir(sdir):
        subject_id = file.split("_")[0]
        if subject_id in subject_ids:
            subject_ids[subject_id] += 1
        else:
            subject_ids[subject_id] = 1

    print(sdir.split("/")[6], len(subject_ids), np.mean(list(subject_ids.values())))
