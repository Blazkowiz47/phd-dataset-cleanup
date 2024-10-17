import torch
from morphs.ladimo.magface import get_model


# Create MagFace model
def get_magface_model(path):
    model = get_model(path)
    model.eval()
    y = model(torch.rand(1, 3, 112, 112))
    print(y.shape)
    return model


get_magface_model("./morphs/models/magface/magface_epoch_00025.pth")
