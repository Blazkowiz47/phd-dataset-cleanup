#!/usr/bin/env python


import torch
import torch.utils.data.distributed

from .network_inf import builder_inf


def get_model(
    ckpt, arch="iresnet100", embedding_size=512, cpu_mode=False
) -> torch.nn.Module:
    return builder_inf(arch, embedding_size, ckpt, cpu_mode)
