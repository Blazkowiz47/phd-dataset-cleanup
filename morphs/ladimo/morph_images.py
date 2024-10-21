import os
import sys
from typing import Any, Tuple

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid
from tqdm import tqdm

from .ldm.models.diffusion.ddim import DDIMSampler
from .ldm.util import instantiate_from_config
from .magface import get_model as get_magface_model
from .slerp import slerp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model_from_config(config, ckpt) -> torch.nn.Module:
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)  # , map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    return model


def get_model() -> Tuple[torch.nn.Module, Any]:
    config_dir = "./morphs/ladimo/configs/latent-diffusion/frgc-ldm-vq-f8.yaml"

    ladimo_model_dir = "./models/logs/2023-11-15T10-04-11_ffhq-ldm-vq-f8/checkpoints/epoch=000096.ckpt"

    config = OmegaConf.load(config_dir)
    model = load_model_from_config(config, ladimo_model_dir)
    return model, config


def get_magface_feature(magface, img_file) -> torch.Tensor:
    img = Image.open(img_file).convert("RGB").resize((112, 112))
    input = torch.tensor(np.array(img)).float()
    input = input.reshape((1, 112, 112, 3))
    input = input.permute(0, 3, 1, 2).contiguous().cuda()
    return magface(input).reshape((1, 1, 512))


def morph(model, magface, img1, img2) -> Any:
    x1 = torch.from_numpy(
        np.array(Image.open(img1).convert("RGB")).reshape((1, 256, 256, 3))
    ).to("cuda")
    x2 = torch.from_numpy(
        np.array(Image.open(img2).convert("RGB")).reshape((1, 256, 256, 3))
    ).to("cuda")

    c1 = get_magface_feature(magface, img1)
    c2 = get_magface_feature(magface, img2)

    # Compute linear interpolation between (this is where thwwwwwwwwwwwe morphing happens)
    cm = slerp(c1, c2, 0.5)
    batch = {"image": x1, "fr_embeds": cm}
    z, c, x, xrec, xc = model.get_input(
        batch,
        "image",
        return_first_stage_outputs=True,
        force_c_encode=True,
        return_original_cond=True,
        bs=1,
    )

    ts = torch.full((1,), 999, device=model.device, dtype=torch.long)
    z_t = model.q_sample(x_start=z, t=ts, noise=None)

    img, progressives = model.progressive_denoising(
        cm, shape=(3, 64, 64), batch_size=1, x_T=z_t, start_T=999, x0=z
    )
    x_morphed = model.decode_first_stage(img)
    x_morphed = rearrange(x_morphed, "b c h w -> b h w c")
    x_stacked = torch.stack([x_morphed]).squeeze()
    x_stacked = (x_stacked + 1.0) / 2.0
    denoise_grid = rearrange(x_stacked, "h w c -> c h w")
    denoise_grid = 255.0 * make_grid(denoise_grid, nrow=1).cpu().numpy()
    denoise_grid = rearrange(denoise_grid, "c h w -> h w c")

    return denoise_grid.astype(np.uint8)


def driver(args: Tuple[int, str, str, str]):
    process_num, src_dir, morph_list_csv, output_dir = args
    with open(morph_list_csv, "r") as fp:
        morph_list = fp.readlines()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, "taming-transformers"))
    sys.path.append(os.path.join(current_dir, "src", "clip"))

    model, _ = get_model()
    _ = DDIMSampler(model)
    magface = get_magface_model("./models/magface/magface_epoch_00025.pth")
    model.eval()
    magface.eval()
    model.cuda()
    magface.cuda()

    for pair in tqdm(morph_list, position=process_num):
        if not pair.strip():
            continue

        splited_pair = pair.strip().split(",")
        img1_path = splited_pair[0]
        img2_path = splited_pair[1]
        img1 = os.path.join(src_dir, img1_path)
        img2 = os.path.join(src_dir, img2_path)
        temp = (
            os.path.split(img1)[1].split(".")[0]
            + "-vs-"
            + os.path.split(img2)[1].split(".")[0]
        )
        output = os.path.join(output_dir, temp + ".png")
        morphed_image = morph(model, magface, img1, img2)
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(morphed_image).save(output)
