from templates import *
from templates_latent import *
import PIL
device = 'cuda:0'
conf = ffhq256_autoenc_latent()
conf.T_eval = 100
conf.latent_T_eval = 100
# print(conf.name)
model = LitModel(conf)
state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
print(model.load_state_dict(state['state_dict'], strict=False))
model.to(device)
torch.manual_seed(4)
imgs = model.sample(8, device=device, T=20, T_latent=200)
for i in range(len(imgs)):
    image = PIL.Image.fromarray(np.uint8(255*imgs[i].cpu().permute([1, 2, 0]).numpy()))
    image.save(str(i)+'.png')
