import sys
from .diffae.templates import *
from torchvision import transforms
from PIL import Image
import torch
from torch.functional import F

path_to_diff_model = "./morphs/mordiff/diffae"

sys.path.append(path_to_diff_model)


def morph_two_images(image1, image2, output):
    device = "cuda"
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load("./models/ffhq/last.ckpt", map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    image_size = 256

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    to_pil_image = transforms.ToPILImage()

    def load_image(path):
        img = Image.open(path)
        img = img.convert("RGB")
        if transform is not None:
            img = transform(img)

        return img

    img1 = load_image(image1)
    img2 = load_image(image2)

    batch = torch.stack([img1, img2])

    cond = model.encode(batch.to(device))

    T = 250
    xT = model.encode_stochastic(batch.to(device), cond, T=T)

    alpha = torch.tensor([0.0, 0.5, 1.0]).to(device)
    intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

    def cos(a, b):
        a = a.view(-1)
        b = b.view(-1)
        a = F.normalize(a, dim=0)
        b = F.normalize(b, dim=0)
        return (a * b).sum()

    theta = torch.arccos(cos(xT[0], xT[1]))
    x_shape = xT[0].shape
    intp_x = (
        torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None]
        + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]
    ) / torch.sin(theta)
    intp_x = intp_x.view(-1, *x_shape)

    pred = model.render(intp_x, intp, T=20)
    pred = to_pil_image(pred[1].cpu())
    pred.save(output)
