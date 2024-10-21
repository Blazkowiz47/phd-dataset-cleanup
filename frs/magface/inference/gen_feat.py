#!/usr/bin/env python


import cv2
from numpy.typing import NDArray
import torch
from torch.nn import Module
from torchvision import transforms
import torch.utils.data.distributed

from .network_inf import builder_inf


def get_model(
    ckpt="./models/frs_models/magface/magface_epoch_00025.pth",
    arch="iresnet100",
    embedding_size=512,
    cpu_mode=False,
) -> Module:
    model = builder_inf(arch, embedding_size, ckpt, cpu_mode)
    model.eval().cuda()
    return model


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]
)


def transform(fname: str) -> torch.Tensor:
    img = cv2.imread(fname)
    inputs = preprocess(img)
    if isinstance(inputs, torch.Tensor):
        return inputs

    raise ValueError("something wrong")


def get_features(fname: str, model: Module) -> NDArray:
    inputs = transform(fname).cuda()
    features = model(inputs)
    features = features.data.detach().cpu().numpy()
    return features.squeeze()
