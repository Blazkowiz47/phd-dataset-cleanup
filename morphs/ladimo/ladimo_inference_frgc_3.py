import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(os.path.join(parent_dir, 'taming-transformers'))
sys.path.append(os.path.join(current_dir,'taming-transformers'))
sys.path.append(os.path.join(current_dir,'src','clip'))
from taming.models import vqgan
from ldm.models.diffusion.ddim import DDIMSampler 
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms
from ldm.util import instantiate_from_config, default
from omegaconf import OmegaConf
from slerp import slerp
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import csv


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model():
    config_dir = os.path.join("/home/ubuntu/cluster/nbl-users/Haoyu/MA_ladimo/latent-diffusion/configs/latent-diffusion/frgc-ldm-vq-f8.yaml")
    ladimo_model_dir = os.path.join("/home/ubuntu/cluster/nbl-users/Haoyu/MA_ladimo/latent-diffusion/logs/2023-11-15T10-04-11_ffhq-ldm-vq-f8/checkpoints/epoch=000096.ckpt")
    config = OmegaConf.load(config_dir)  
    model = load_model_from_config(config, ladimo_model_dir)
    return model, config


print("Load Latent Diffusion Model")
model, config = get_model()
sampler = DDIMSampler(model)

print("Load and prepare Inference data")
ldm_data = instantiate_from_config(config.data)
ldm_data.prepare_data()
ldm_data.setup()

# NOTE: Morph two example identities -> remember to build your pairing individually

morph_list_path = './Morph_LADIMO_P2_Female.txt'
morph_list = []
with open(morph_list_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        morph_list.append(row)

error_list = []
error_log_path = './error_log.txt'

reference_dir = './taming-transformers/taming/data/morph_db/reference/'
reference_list = os.listdir(reference_dir)

assert len(ldm_data.datasets['train']) == len(reference_list)

target_dir = './FRGC_LADIMO/ICAO_P2/Female/'
os.makedirs(target_dir, exist_ok=True)

for morph_pair in morph_list:
    
    first_morph_idx = reference_list.index(morph_pair[0])
    second_morph_idx = reference_list.index(morph_pair[1])

    x1 = torch.from_numpy(ldm_data.datasets['train'][first_morph_idx]['image'].reshape((1, 256, 256, 3))).to("cuda")
    x2 = torch.from_numpy(ldm_data.datasets['train'][second_morph_idx]['image'].reshape((1, 256, 256, 3))).to("cuda")

    c1 = ldm_data.datasets['train'][first_morph_idx]['fr_embeds'].reshape((1, 1, 512)).to("cuda")
    c2 = ldm_data.datasets['train'][second_morph_idx]['fr_embeds'].reshape((1, 1, 512)).to("cuda")

    # Compute linear interpolation between (this is where thwwwwwwwwwwwe morphing happens)
    cm = slerp(c1, c2, 0.5)
    batch =  {'image': x1, 'fr_embeds': cm}
    z, c, x, xrec, xc = model.get_input(batch, 'image',
                                    return_first_stage_outputs=True,
                                    force_c_encode=True,
                                    return_original_cond=True,
                                    bs=1)

    ts = torch.full((1,), 999, device=model.device, dtype=torch.long)
    z_t = model.q_sample(x_start = z, t = ts, noise = None)

    img, progressives = model.progressive_denoising(cm, shape=(3, 64, 64), batch_size=1, x_T = z_t, start_T=999, x0 = z)
    x_morphed = model.decode_first_stage(img)
    x_morphed = rearrange(x_morphed, 'b c h w -> b h w c')
    x_stacked = torch.stack([x_morphed]).squeeze()
    x_stacked = (x_stacked + 1.0) / 2.0
    denoise_grid = rearrange(x_stacked, 'h w c -> c h w')
    denoise_grid = 255. * make_grid(denoise_grid, nrow=1).cpu().numpy()
    denoise_grid = rearrange(denoise_grid, 'c h w -> h w c')
    fname = 'MLDM_'+morph_pair[0].split('.')[0]+'_'+morph_pair[1].split('.')[0]+'_W0.50_B0.50_AR_CE.png'
    outdir = os.path.join(target_dir, fname)
    Image.fromarray(denoise_grid.astype(np.uint8)).save(outdir)