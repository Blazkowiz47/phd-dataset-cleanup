from model import Backbone
import torch
from PIL import Image
from mtcnn import MTCNN

from torchvision.transforms import Compose, ToTensor, Normalize

mtcnn = MTCNN()


def get_img(img_path, device):
    img = Image.open(img_path)
    face = mtcnn.align(img)
    transfroms = Compose(
        [ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transfroms(face).to(device).unsqueeze(0)


device = 'cuda'
img1 = get_img('face1.png', device)
img2 = get_img('face2.png', device)
img3 = get_img('face3.png', device)

print(img1.shape)

model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
model.load_state_dict(torch.load('model_ir_se50.pth'))
model.eval()
model.to(device)

emb1 = model(img1)[0]
emb2 = model(img2)[0]
emb3 = model(img3)[0]
print(emb1.shape)

sim_12 = emb1.dot(emb2).item()
sim_13 = emb1.dot(emb3).item()

print(sim_12)
print(sim_13)