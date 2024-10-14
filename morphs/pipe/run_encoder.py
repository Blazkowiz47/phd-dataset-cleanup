import sys
import os

path_to_diff_model = "./"
suffix_list = ['png','jpg','jpeg','PNG','JPG','JPEG']

sys.path.append(path_to_diff_model)

from templates import *
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torch.functional import F
from tqdm import tqdm
# import matplotlib.pyplot as plt

from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--source_dir",
                        type=str,
                        help="Directory of source images")
    parser.add_argument("--output",
                        type=str,
                        help="output path")

    args = parser.parse_args()

    # load the model
    device = 'cuda:0'
    conf = ffhq256_autoenc()
    # print(conf.name)
    model = LitModel(conf)
    state = torch.load(f'{path_to_diff_model}/checkpoints/{conf.name}/last.ckpt', map_location=device)
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # images to fuse: needs to be aligned
    source_dir = args.source_dir
    outfolder = args.output
    os.makedirs(outfolder,exist_ok=True)
    image_size = conf.img_size # 256

    source_list = os.listdir(source_dir)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    to_pil_image = transforms.ToPILImage()

    def load_image(path):
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('RGB')

        assert img.size[0] == img.size[1]

        if transform is not None:
            img = transform(img)

        return img

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for i in tqdm(range(len(source_list))):
        image = source_list[i]
        if image.split('.')[-1] in suffix_list:

            img_path = os.path.join(source_dir,image)
            img = load_image(img_path)

            batch = torch.stack([
                img 
            ])

            cond = model.encode(batch.to(device))

            cond = cond[0].cpu().detach().numpy()
            # print(cond[0].cpu().detach().numpy().shape)

            output_path = os.path.join(outfolder,image[:-3]+'npy')
            np.save(output_path,cond)
