import numpy as np
import torch
import cv2
from numpy.typing import NDArray
from .backbone import iresnet100


def get_model(
    ckpt: str = "./models/frs_models/arcface/ms1mv3_arcface_r100_fp16/backbone.pth",
):
    model = iresnet100(fp16=True)
    model.load_state_dict(torch.load(ckpt))
    model.eval().cuda()
    return model


def transform(fname: str) -> NDArray:
    img = cv2.imread(fname)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    return img.cuda()


def get_features(fname: str, model) -> NDArray:
    img = transform(fname)
    features = model(img)
    return features.squeeze().detach().cpu()


if __name__ == "__main__":
    model = get_model(
        "/home/ubuntu/phd-dataset-cleanup/models/frs_models/arcface/ms1mv3_arcface_r100_fp16/backbone.pth"
    )
    imgname = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/frgc/digital/bonafide/test/02463_1.jpg"
    img = transform(imgname)
    print(img.shape)
    features = model(img)
    print(features.shape)
