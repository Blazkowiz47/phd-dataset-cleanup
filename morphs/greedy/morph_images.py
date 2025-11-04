import os

import lpips
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from . import iresnet
from .scheduler import DPMSolverMultiStepScheduler
from .templates import *


class MorphDataset(torch.utils.data.Dataset):
    def __init__(
        self, src_dir, morph_list_csv, outdir, image_size, return_opencv=False
    ):
        self.src_dir = src_dir
        self.morph_list_csv = morph_list_csv
        self.outdir = outdir

        self.image_size = image_size
        self.files: List[Tuple[str, str, str]] = []
        print("Building morph dataset for greedy")
        with open(morph_list_csv, "r") as fp:
            pairs = fp.readlines()
        
        done = 0
        for pair in pairs:
            if not pair.strip():
                continue
            if len(pair.split(",")) != 2:
                continue
            img1, img2 = pair.split(",")
            img1, img2 = img1.strip(), img2.strip()

            fname = img1.split(".")[0] + "-vs-" + img2.split(".")[0] + ".png"
            img1, img2 = os.path.join(src_dir, img1), os.path.join(src_dir, img2)
            if not os.path.exists(img1):
                img1 = img1.replace(".png", ".jpg")
                img1 = os.path.join(os.path.split(img1)[0], os.path.split(img1)[1])
                if not os.path.exists(img1):
                    img1 = os.path.join(
                        os.path.split(img1)[0], os.path.split(img1)[1].replace("_", "-")
                    )
                    if not os.path.exists(img1):
                        continue

            if not os.path.exists(img2):
                img2 = img2.replace(".png", ".jpg")
                img2 = os.path.join(os.path.split(img2)[0], os.path.split(img2)[1])
                if not os.path.exists(img2):
                    img2 = os.path.join(
                        os.path.split(img2)[0], os.path.split(img2)[1].replace("_", "-")
                    )
                    if not os.path.exists(img2):
                        continue

            fname = os.path.join(outdir, fname)
            os.makedirs(outdir, exist_ok=True)
            if os.path.isfile(fname):
                continue
            self.files.append((img1, img2, fname))
            done += 1
            # if done == 50:
            #     break

        transform = [
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)
        print("Initialised dataset")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path_a, path_b, dest = self.files[index]

        img = Image.open(path_a)
        img = img.convert("RGB")
        img_a = self.transform(img)

        img = Image.open(path_b)
        img = img.convert("RGB")
        img_b = self.transform(img)

        return img_a, img_b, dest


##############################################################################
#
# Code for identity loss functions
#
##############################################################################


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)


class MorphLPIPS(lpips.LPIPS):
    def morph_loss(self, morph, in0, in1, retPerLayer=False, normalize=False):
        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
            morph = 2 * morph - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == "0.1"
            else (in0, in1)
        )
        morph_input = self.scaling_layer(morph) if self.version == "0.1" else morph
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        outs_morph = self.net.forward(morph_input)
        feats0, feats1, diffs = {}, {}, {}
        feats_morph = {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = (
                lpips.normalize_tensor(outs0[kk]),
                lpips.normalize_tensor(outs1[kk]),
            )
            feats_morph[kk] = lpips.normalize_tensor(outs_morph[kk])
            ideal_morph = 0.5 * (feats0[kk] + feats1[kk])
            # diffs[kk] = (feats0[kk]-feats1[kk])**2
            diffs[kk] = (ideal_morph - feats_morph[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [
                    upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.L)
                ]
        else:
            if self.spatial:
                res = [
                    upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True)
                    for kk in range(self.L)
                ]

        val = 0
        for l in range(self.L):
            val += res[l]

        if retPerLayer:
            return (val, res)
        else:
            return val


# Create ArcFace model
def get_arcface_model(path):
    model = iresnet.IResNet(iresnet.IBasicBlock, [3, 13, 30, 3], fp16=True)
    model.load_state_dict(torch.load(path))

    return model


def arcface_preprocess(x):
    x = F.interpolate(x, size=112, mode="bilinear")

    return x


# Morph-PIPE identity prior Zhang et al.
def zhang_identity_prior(model, x_morph, x_a, x_b, weight_factor=0.5):
    x_morph = arcface_preprocess(x_morph)
    x_a = arcface_preprocess(x_a)
    x_b = arcface_preprocess(x_b)

    # Get embeddings
    m_embed = model(x_morph)
    a_embed = model(x_a)
    b_embed = model(x_b)

    # Loss ID
    loss_id = (
        1.0
        - F.cosine_similarity(m_embed, a_embed, dim=1)
        + 1.0
        - F.cosine_similarity(m_embed, b_embed, dim=1)
    )
    loss_id_diff = torch.abs(
        F.cosine_similarity(m_embed, a_embed, dim=1)
        - F.cosine_similarity(m_embed, b_embed, dim=1)
    )

    return weight_factor * loss_id + (1.0 - weight_factor) * loss_id_diff


def worst_case_loss(model, x_morph, x_a, x_b, dist="l2"):
    x_morph = arcface_preprocess(x_morph)
    x_a = arcface_preprocess(x_a)
    x_b = arcface_preprocess(x_b)

    # Get embeddings
    m_embed = model(x_morph)
    a_embed = model(x_a)
    b_embed = model(x_b)

    if dist == "l2":
        worst_case_embed = 0.5 * (a_embed + b_embed)
    elif dist == "cosine":
        worst_case_embed = (a_embed + b_embed) / F.normalize(a_embed + b_embed, dim=1)
    else:
        raise Exception("Invalid option for argument `dist` was found", dist)

    return 1.0 - F.cosine_similarity(m_embed, worst_case_embed, dim=1)


##############################################################################
#
# Some small helper functions.
#
##############################################################################

rescale_img = lambda x: (x + 1.0) / 2.0


def slerp(z1, z2, alpha):
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))

    return (
        torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
        + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )


##############################################################################
#
# Numerical ODE Solvers for Probability Flow ODE includes Greedy Solver
#
##############################################################################


@torch.no_grad()
def solve_pf_ode(model, scheduler, xt, z, noise_level=1.0, device=None):
    """
    Solves the PF ODE with a numerical ODE solver.

    Args:
        model (nn.Module): Noise prediction U-Net (or x-prediction / v-prediction).
        scheduler (DPMSolverMultiStepScheduler): An instance of the noise scheduler and k-th order ODE solver.
        xt (torch.Tensor): Morphed noisy image at time `t`.
        z (torch.Tensor): Latent representation of `xt`.
        noise_level (float, optional): Amount of noise, bijective mapping to timestep. Defaults to 1.
        device (torch.device, optional): Desired device of the returned tensor. Defaults to None.

    Returns:
        x0 (torch.Tensor): The denoised image.
    """

    device = device if device is not None else next(model.ema_model.parameters()).device
    cond = z

    b = xt.shape[0]
    x = xt

    timesteps = scheduler.timesteps

    scheduler.reset_sampler()

    if noise_level < 1.0:
        timesteps = timesteps[int((1.0 - noise_level) * len(timesteps)) :]

    for i, t in enumerate(
        tqdm(timesteps, desc="Sampling loop...", total=len(timesteps))
    ):
        t_batch = torch.ones(b, device=device) * t

        model_output = model.ema_model.forward(x=x, t=t_batch, cond=cond).pred
        x = scheduler.step(model_output, t, x)

    return rescale_img(x)


@torch.no_grad()
def encode_diffae(model, x0, z, T=None, noise_level=1.0, device=None):
    """
    Solves the PF ODE as time runs forwards using the unique formulation from the DiffAE paper https://arxiv.org/abs/2111.15640

    Args:
        model (nn.Module): Noise prediction U-Net (or x-prediction / v-prediction).
        x0 (torch.Tensor): Initial image.
        z (torch.Tensor): Latent representation of `x0`.
        T (int, optional): Number of sampling steps. Defaults to None.
        noise_level (float, optional): Amount of noise, bijective mapping to timestep. Defaults to 1.
        device (torch.device, optional): Desired device of the returned tensor. Defaults to None.

    Returns:
        xt (torch.Tensor): The solution to the PF ODE at time `t` given `x0`
    """

    sampler = (
        model.eval_sampler
        if T is None
        else model.conf._make_diffusion_conf(T).make_sampler()
    )
    device = device if device is not None else next(model.ema_model.parameters()).device

    stop_step = int(sampler.num_timesteps * noise_level)
    sample = x0

    for i in tqdm(range(stop_step), desc="Encoding stochastic code"):
        t = torch.tensor([i] * sample.shape[0], device=device)

        sample = sampler.ddim_reverse_sample(
            model.ema_model, sample, t=t, model_kwargs={"cond": z}
        )["sample"]

    return sample


@torch.no_grad()
def encode_pf_ode(model, scheduler, x0, z, noise_level=1.0, device=None):
    """
    Solves the PF ODE as time runs forwards with a numerical ODE solver.

    Args:
        model (nn.Module): Noise prediction U-Net (or x-prediction / v-prediction).
        scheduler (DPMSolverMultiStepScheduler): An instance of the noise scheduler and k-th order ODE solver.
        x0 (torch.Tensor): Initial image.
        z (torch.Tensor): Latent representation of `x0`.
        noise_level (float, optional): Amount of noise, bijective mapping to timestep. Defaults to 1.
        device (torch.device, optional): Desired device of the returned tensor. Defaults to None.

    Returns:
        xt (torch.Tensor): The solution to the PF ODE at time `t` given `x0`
    """

    device = device if device is not None else next(model.ema_model.parameters()).device

    timesteps = torch.flip(scheduler.timesteps, dims=(0,))
    scheduler.reset_sampler()

    x = x0

    if noise_level < 1.0:
        timesteps = timesteps[: int(noise_level * len(timesteps))]

    for t in tqdm(timesteps, desc="Encoding stochastic code", total=len(timesteps)):
        t_batch = torch.ones(x.shape[0], device=device) * t

        model_output = model.ema_model.forward(x=x, t=t_batch, cond=z).pred
        x = scheduler.reverse_step(model_output, t, x)

    return x


@torch.no_grad()
def greedy_search_pf_ode(
    model,
    scheduler,
    xt,
    z_a,
    z_b,
    x_a,
    x_b,
    loss_fn,
    n_weights=21,
    noise_level=1.0,
    device=None,
):
    """
    Greedy-DiM-S algorithm. See Eq. (10) in https://arxiv.org/abs/2404.06025.

    Args:
        model (nn.Module): Noise prediction U-Net (or x-prediction / v-prediction).
        scheduler (DPMSolverMultiStepScheduler): An instance of the noise scheduler and k-th order ODE solver.
        xt (torch.Tensor): Morphed noisy image at time `t`.
        z_a (torch.Tensor): Latent representation of `x_a`.
        z_b (torch.Tensor): Latent representation of `x_b`.
        x_a (torch.Tensor): Bona fide image of identity a.
        x_b (torch.Tensor): Bona fide image of identity b.
        loss_fn (func): Loss function to guide generation.
        n_weights (int, optional): Size of search space. Defaults to 21.
        noise_level (float, optional): Amount of noise, bijective mapping to timestep. Defaults to 1.
        device (torch.device, optional): Desired device of the returned tensor. Defaults to None.

    Returns:
        x_morph (torch.Tensor): The morphed image.
    """

    device = device if device is not None else next(model.ema_model.parameters()).device

    b = xt.shape[0]

    timesteps = scheduler.timesteps

    scheduler.reset_sampler()

    if noise_level < 1.0:
        timesteps = timesteps[int((1.0 - noise_level) * len(timesteps)) :]

    for i, t in enumerate(
        tqdm(timesteps, desc="Sampling loop...", total=len(timesteps))
    ):
        t_batch = torch.ones(b, device=device) * t

        out_a = model.ema_model.forward(x=xt, t=t_batch, cond=z_a).pred
        out_b = model.ema_model.forward(x=xt, t=t_batch, cond=z_b).pred

        alphas = torch.linspace(0.0, 1.0, n_weights, device=device)
        best_loss = torch.ones(b, dtype=torch.float, device=device) * 1e4
        best_out = out_a

        progress_bar = tqdm(total=n_weights)
        progress_bar.set_description("Searching for optimal alpha...")

        for alpha in alphas:
            out = slerp(out_a, out_b, alpha)
            x0_pred = scheduler.convert_model_output(out, t, xt)

            with torch.no_grad():
                loss = loss_fn(x0_pred, x_a, x_b)

            do_update = (loss < best_loss).float()
            best_loss = do_update * loss + (1.0 - do_update) * best_loss
            best_out = (
                do_update[:, None, None, None] * out
                + (1.0 - do_update)[:, None, None, None] * best_out
            )

            progress_bar.update(1)
            logs = {"best_loss": best_loss.mean().item()}
            progress_bar.set_postfix(**logs)

        out = best_out

        xt = scheduler.step(out, t, xt)

    return rescale_img(xt)


def greedy_solve_pf_ode(
    model,
    scheduler,
    xt,
    z_morph,
    x_a,
    x_b,
    loss_fn,
    n_opt_steps=50,
    opt_stride=1,
    noise_level=1.0,
    opt_kwargs={},
    device=None,
):
    """
    Greedy-DiM* algorithm. See Algorithm 1 in https://arxiv.org/abs/2404.06025.

    Args:
        model (nn.Module): Noise prediction U-Net (or x-prediction / v-prediction).
        scheduler (DPMSolverMultiStepScheduler): An instance of the noise scheduler and k-th order ODE solver.
        xt (torch.Tensor): Morphed noisy image at time `t`.
        z_morph (torch.Tensor): Morphed latent representation.
        x_a (torch.Tensor): Bona fide image of identity a.
        x_b (torch.Tensor): Bona fide image of identity b.
        loss_fn (func): Loss function to guide generation.
        n_opt_steps (int, optional): Number of optimization steps per greedy timestep of the ODE solver. Defaults to 50.
        opt_stride (int, optional): Stride for greedy strategy. Defaults to 1.
        noise_level (float, optional): Amount of noise, bijective mapping to timestep. Defaults to 1.
        opt_kwargs (dict, optional): Dictionary of optimizer arguments. Defaults to {}.
        device (torch.device, optional): Desired device of the returned tensor. Defaults to None.

    Returns:
        x_morph (torch.Tensor): The morphed image.
    """
    device = device if device is not None else next(model.ema_model.parameters()).device

    b = xt.shape[0]

    timesteps = scheduler.timesteps
    scheduler.reset_sampler()

    if noise_level < 1.0:
        timesteps = timesteps[int((1.0 - noise_level) * len(timesteps)) :]

    for i, t in enumerate(
        tqdm(timesteps, desc="Sampling loop...", total=len(timesteps))
    ):
        t_batch = torch.ones(b, device=device) * t

        with torch.no_grad():
            out = model.ema_model.forward(x=xt, t=t_batch, cond=z_morph).pred

        if (i % opt_stride) == 0:
            out = out.detach().clone().requires_grad_(True)
            opt = torch.optim.RAdam([out], **opt_kwargs)

            x0_pred = scheduler.convert_model_output(out, t.detach(), xt.detach())
            best_loss = loss_fn(x0_pred, x_a.detach(), x_b.detach())

            best_out = out

            progress_bar = tqdm(total=n_opt_steps)
            progress_bar.set_description("Optimizing Latents...")

            for _ in range(n_opt_steps):
                opt.zero_grad()

                x0_pred = scheduler.convert_model_output(out, t.detach(), xt.detach())
                loss = loss_fn(x0_pred, x_a.detach(), x_b.detach())

                loss.mean().backward()
                opt.step()

                # Update per sample not per batch
                do_update = (loss < best_loss).float()
                best_loss = do_update * loss + (1.0 - do_update) * best_loss
                best_out = (
                    do_update[:, None, None, None] * out
                    + (1.0 - do_update)[:, None, None, None] * best_out
                )

                progress_bar.update(1)
                logs = {"loss": loss.mean().item()}
                progress_bar.set_postfix(**logs)

            out = best_out
            print(f"greedy loss at step {t} is {best_loss.mean().item():.4f}")

        xt = scheduler.step(out, t, xt)

    return rescale_img(xt)


def driver(args: Tuple[int, str, str, str]) -> None:
    process_num, src_dir, morph_list_csv, outdir = args
    config = "./morphs/greedy/configs/greedy_dim.yml"
    batch_size = 16
    print("Driver start")

    with open(config, "r") as f:
        config = yaml.safe_load(f)
    print("loaded config")

    device = "cuda"
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(
        f"./models/checkpoints/{conf.name}/last.ckpt", map_location="cpu"
    )
    model.load_state_dict(state["state_dict"], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    scheduler = DPMSolverMultiStepScheduler(**config["scheduler_kwargs"])
    scheduler.set_timesteps(config["sampling_timesteps"])
    rev_scheduler = None

    if config["encoding_solver"] == "dpmsolver":
        rev_scheduler = DPMSolverMultiStepScheduler(**config["encoder_kwargs"])
        rev_scheduler.set_timesteps(config["encoding_timesteps"])

    # Create loss function
    loss_fn = None

    if "greedy" in config:
        loss_model = get_arcface_model(
            "./models/glint360k_cosface_r100_fp16_0.1/backbone.pth"
        )
        loss_model.eval().to(device)

        if config["loss_fn"]["type"] == "zhang_identity_prior":
            loss_fn = lambda x, y, z: zhang_identity_prior(loss_model, x, y, z)
        elif config["loss_fn"]["type"] == "worst_case":
            loss_fn = lambda x, y, z: worst_case_loss(
                loss_model, x, y, z, dist=config["loss_fn"]["dist"]
            )

        else:
            raise Exception("Invalid `loss_fn` config!")

    print("loaded model")
    dataset = MorphDataset(src_dir, morph_list_csv, outdir, conf.img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    print("loaded dataset")

    for img_a, img_b, output_path in tqdm(dataloader):
        img_a = img_a.to(device)
        img_b = img_b.to(device)

        batch_size = img_a.shape[0]

        with torch.no_grad():
            z_a = model.encode(img_a)
            z_b = model.encode(img_b)

        if config["encoding_solver"] == "diffae":
            xs = encode_diffae(
                model,
                torch.cat((img_a, img_b), dim=0),
                torch.cat((z_a, z_b), dim=0),
                T=config["encoding_timesteps"],
            )
        elif config["encoding_solver"] == "dpmsolver":
            xs = encode_pf_ode(
                model,
                rev_scheduler,
                torch.cat((img_a, img_b), dim=0),
                torch.cat((z_a, z_b), dim=0),
            )
        else:
            raise Exception("Invalid value for `encoding_solver`")

        x_a, x_b = xs.chunk(2, dim=0)
        x_morph = slerp(x_a, x_b, 0.5)
        z_morph = torch.lerp(z_a, z_b, 0.5)

        if "greedy" in config:
            if config["greedy"]["type"] == "opt":
                morph = greedy_solve_pf_ode(
                    model,
                    scheduler,
                    x_morph,
                    z_morph,
                    img_a,
                    img_b,
                    loss_fn,
                    **config["greedy"]["kwargs"],
                )
            elif config["greedy"]["type"] == "search":
                morph = greedy_search_pf_ode(
                    model,
                    scheduler,
                    x_morph,
                    z_a,
                    z_b,
                    img_a,
                    img_b,
                    loss_fn,
                    **config["greedy"]["kwargs"],
                )
            else:
                raise Exception("Invalid type for greedy algorithm.")
        else:
            morph = solve_pf_ode(model, scheduler, x_morph, z_morph)

        for i in range(batch_size):
            torchvision.utils.save_image(morph[i], output_path[i])
