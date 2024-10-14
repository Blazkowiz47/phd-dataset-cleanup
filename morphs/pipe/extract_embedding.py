from model import Backbone
import torch
from PIL import Image
from mtcnn import MTCNN
import os
from tqdm import tqdm
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize

mtcnn = MTCNN()


def get_img(img_path, device):
    img = Image.open(img_path)
    face = mtcnn.align(img)
    transfroms = Compose(
        [ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transfroms(face).to(device).unsqueeze(0)

suffix_list = ['png','jpg','jpeg','PNG','JPG','JPEG']
source_dir = '../../Dataset/FFHQ/FFHQ_sampled'
device = 'cuda'
filename_list = os.listdir(source_dir)
save_path = '../data_embd/'
os.makedirs(save_path,exist_ok=True)

model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
model.load_state_dict(torch.load('model_ir_se50.pth'))
model.eval()
model.to(device)


filenames = []
embds = []
for i in tqdm(range(len(filename_list))):
    filename = filename_list[i]

    if filename.split('.')[-1] in suffix_list:

        file_path = os.path.join(source_dir,filename)
        try:
            img = get_img(file_path, device)
        except:
            continue

        emb = model(img)[0].cpu().detach().numpy()

        filenames.append(filename)
        embds.append(emb)

        save_file_path = os.path.join(save_path,filename[:-3]+'npy') 
        np.save(save_file_path,emb)

# embds_dict = dict(zip(filenames, list(embds)))
# np.save(os.path.join(save_path,'embds_arcface.npy'),embds_dict)


