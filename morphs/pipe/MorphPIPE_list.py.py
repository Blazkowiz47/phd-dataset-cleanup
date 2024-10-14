import sys
import os

path_to_diff_model = "./"

sys.path.append(path_to_diff_model)

from templates import *
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from torch.functional import F
from tqdm import tqdm
from scipy.spatial.distance import cosine
# import matplotlib.pyplot as plt
from model_arcface import Backbone
from mtcnn import MTCNN
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--morph_list",
                        type=str,
                        help="list of morph pairs")
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

    mtcnn = MTCNN()
    arcface_model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    arcface_model.load_state_dict(torch.load('./InsightFace_Pytorch/model_ir_se50.pth'))
    arcface_model.eval()
    arcface_model.to(device)


    # images to fuse: needs to be aligned
    morph_list_path = args.morph_list
    outfolder = args.output

    image_size = conf.img_size # 256

    morph_list = []
    with open(morph_list_path,'r') as f:
        morph_list = f.readlines()

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

    def load_image_arcface(img_path, device):
        img = Image.open(img_path)
        face = mtcnn.align(img)
        transfroms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        return transfroms(face).to(device).unsqueeze(0)

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for i in tqdm(range(len(morph_list))):
        pair = morph_list[i]

        splited_pair = pair.split(',')
        img1_path = splited_pair[0]
        img2_path = splited_pair[1][:-1]# remove '\n'
        img1 = load_image(img1_path)
        img2 = load_image(img2_path) 

        batch = torch.stack([
            img1, 
            img2
        ])

        cond = model.encode(batch.to(device))

        T = 250
        xT = model.encode_stochastic(batch.to(device), cond, T=T)

        alpha = torch.tensor([0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]).to(cond.device)
        intp = cond[0][None] * (1 - alpha[:, None]) + cond[1][None] * alpha[:, None]

        def cos(a, b):
            a = a.view(-1)
            b = b.view(-1)
            a = F.normalize(a, dim=0)
            b = F.normalize(b, dim=0)
            return (a * b).sum()

        theta = torch.arccos(cos(xT[0], xT[1]))
        x_shape = xT[0].shape
        intp_x = (torch.sin((1 - alpha[:, None]) * theta) * xT[0].flatten(0, 2)[None] + torch.sin(alpha[:, None] * theta) * xT[1].flatten(0, 2)[None]) / torch.sin(theta)
        intp_x = intp_x.view(-1, *x_shape)

        pred = model.render(intp_x, intp, T=20)

        # # torch.manual_seed(1)
        # #fig, ax = plt.subplots(1, 10, figsize=(5*10, 5))
        # fig, ax = plt.subplots(1, 3, figsize=(5*3, 5))
        # for i in range(len(alpha)):
        #     ax[i].imshow(pred[i].permute(1, 2, 0).cpu())

        name1 = img1_path.split("/")[-1].split(".")[0]
        name2 = img2_path.split("/")[-1].split(".")[0]

        # plt.savefig(os.path.join(outfolder, f"comparison_{name1}_and_{name2}.png"))

        # plt.close("all")

        img1_mtcnn = load_image_arcface(img1_path, device)
        img2_mtcnn =load_image_arcface(img2_path, device)
        emb1 = arcface_model(img1_mtcnn)[0].cpu().detach().numpy()
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = arcface_model(img2_mtcnn)[0].cpu().detach().numpy()   
        emb2 = emb2 / np.linalg.norm(emb2)
        
        arcfce_transfroms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        lowest_dist = 99999
        lowest_i = 0
        for i in range(len(pred)):
            img = to_pil_image(pred[i].cpu())
            img = img.convert('RGB')
            
            face = mtcnn.align(img)
            face_mtcnn = arcfce_transfroms(face).to(device).unsqueeze(0)
            embm = arcface_model(face_mtcnn)[0].cpu().detach().numpy()
            embm = embm / np.linalg.norm(embm)

            dist = abs(cosine(embm,emb1)) + abs(cosine(embm,emb2)) + abs(abs(cosine(embm,emb1)) -abs(cosine(embm,emb2)))
            if dist < lowest_dist:
                lowest_dist = dist
                lowest_i = i

        img = to_pil_image(pred[lowest_i].cpu())
        img = img.convert('RGB')
        img.save(os.path.join(outfolder, f"PIPE_{name1}_{name2}.png"))