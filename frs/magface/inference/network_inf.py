#!/usr/bin/env python
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from ..models import iresnet


def load_features(arch, embedding_size):
    if arch == "iresnet34":
        features = iresnet.iresnet34(
            pretrained=False,
            num_classes=embedding_size,
        )
    elif arch == "iresnet18":
        features = iresnet.iresnet18(
            pretrained=False,
            num_classes=embedding_size,
        )
    elif arch == "iresnet50":
        features = iresnet.iresnet50(
            pretrained=False,
            num_classes=embedding_size,
        )
    elif arch == "iresnet100":
        features = iresnet.iresnet100(
            pretrained=False,
            num_classes=embedding_size,
        )
    else:
        raise ValueError()
    return features


class NetworkBuilder_inf(nn.Module):
    def __init__(self, arch, embedding_size):
        super(NetworkBuilder_inf, self).__init__()
        self.features = load_features(arch, embedding_size)

    def forward(self, input):
        # add Fp, a pose feature
        x = self.features(input)
        return x


def load_dict_inf(ckpt, cpu_mode, model):
    if os.path.isfile(ckpt):
        #         print("=> loading pth from {} ...".format(ckpt))
        if cpu_mode:
            checkpoint = torch.load(
                ckpt, weights_only=True, map_location=torch.device("cpu")
            )
        else:
            checkpoint = torch.load(ckpt, weights_only=False)
        _state_dict = clean_dict_inf(model, checkpoint["state_dict"])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(ckpt))
    return model


def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        # # assert k[0:1] == 'features.module.'
        new_k = "features." + ".".join(k.split(".")[2:])
        if (
            new_k in model.state_dict().keys()
            and v.size() == model.state_dict()[new_k].size()
        ):
            _state_dict[new_k] = v
        # assert k[0:1] == 'module.features.'
        new_kk = ".".join(k.split(".")[1:])
        if (
            new_kk in model.state_dict().keys()
            and v.size() == model.state_dict()[new_kk].size()
        ):
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        sys.exit(
            "=> Not all weights loaded, model params: {}, loaded params: {}".format(
                num_model, num_ckpt
            )
        )
    return _state_dict


def builder_inf(arch, embedding_size, ckpt, cpu_mode):
    model = NetworkBuilder_inf(arch, embedding_size)
    # Used to run inference
    model = load_dict_inf(ckpt, cpu_mode, model)
    return model
