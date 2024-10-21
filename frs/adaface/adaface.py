from numpy.typing import NDArray
import numpy as np
from PIL import Image
import torch
from torch.nn import Module
import numpy as np
from . import net


def get_model(
    ckpt: str = "./models/frs_models/adaface/adaface_ir101_webface12m.ckpt",
    architecture="ir_100",
) -> Module:
    # load model and pretrained statedict
    model = net.build_model(architecture)
    statedict = torch.load(ckpt)["state_dict"]
    model_statedict = {
        key[6:]: val for key, val in statedict.items() if key.startswith("model.")
    }
    model.load_state_dict(model_statedict)
    model.eval().cuda()
    return model


def transform(fname: str) -> torch.Tensor:
    pil_rgb_image = Image.open(fname)
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


def get_features(fname: str, model: Module) -> NDArray:
    input = transform(fname)
    features, _ = model(input)
    return features.detach().cpu().numpy().squeeze()
