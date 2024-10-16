import os
import numpy as np
import pandas as pd
import albumentations
from torch.utils.data import Dataset
from torch import randn
import torch

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class FacesBase(Dataset):
    def __init__(self, config = None, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None
        self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex


class CelebAHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/celebahq"
        with open("data/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = NumpyPaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/mnt2/PhD-Marcel/ldm-face-manipulation/latent-diffusion/taming-transformers/data/ffhq"
        img_path = os.path.join(root, "ffhq-train")
        img_names = os.listdir(img_path)
        paths = [os.path.join(root, f"ffhq-train/{relpath}") for relpath in img_names
                 if relpath.endswith(".png")]
        create_path = lambda x: os.path.join(root, f"ffhq-train-magface/{x}")
        shape_code_paths = [f"{create_path(relpath).split('.')[0]}.pt" for relpath in img_names if relpath.endswith(".png")]
        self.labels = {'fr_embeds':  [torch.load(shape_code_path).to('cpu') for shape_code_path in shape_code_paths]}
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=self.labels)


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/mnt2/PhD-Marcel/ldm-face-manipulation/latent-diffusion/taming-transformers/data/ffhq"
        img_path = os.path.join(root, "ffhq-val")
        img_names = os.listdir(img_path)
        paths = [os.path.join(root, f"ffhq-val/{relpath}") for relpath in img_names
                 if relpath.endswith(".png")]
        create_path = lambda x: os.path.join(root, f"ffhq-val-magface/{x}")
        shape_code_paths = [f"{create_path(relpath).split('.')[0]}.pt" for relpath in img_names if relpath.endswith(".png")]
        self.labels = {'fr_embeds':  [torch.load(shape_code_path).to('cpu') for shape_code_path in shape_code_paths]}
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=self.labels)
        
class FRGCTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/mnt2/datasets/face/Marcel-CustomDBs/morph_db"
        meta_data = pd.read_csv(os.path.join(root, "data.csv"))
        relpaths = meta_data['Name']
        paths = [os.path.join(root, f"reference/{relpath}") for relpath in relpaths
                 if relpath.endswith(".png")]
        create_path = lambda x: os.path.join(root, f"reference_magface/{x}")
        shape_code_paths = [f"{create_path(relpath).split('.')[0]}.pt" for relpath in relpaths if relpath.endswith(".png")]
        self.labels = {'fr_embeds':  [torch.load(shape_code_path).to('cpu') for shape_code_path in shape_code_paths]}
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=self.labels)
        

class FRGCValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/mnt2/datasets/face/Marcel-CustomDBs/morph_db"
        meta_data = pd.read_csv(os.path.join(root, "data.csv"))
        relpaths = meta_data['Name']
        paths = [os.path.join(root, f"reference/{relpath}") for relpath in relpaths
                 if relpath.endswith(".png")]
        create_path = lambda x: os.path.join(root, f"reference_magface/{x}")
        shape_code_paths = [f"{create_path(relpath).split('.')[0]}.pt" for relpath in relpaths if relpath.endswith(".png")]
        self.labels = {'fr_embeds':  [torch.load(shape_code_path).to('cpu') for shape_code_path in shape_code_paths]}
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=self.labels)
        
class LADIMOInferenceData(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        fdir = os.path.dirname(__file__)
        root = os.path.join(fdir, "morph_db")
        img_fnames = os.listdir(os.path.join(root, "reference"))
        paths = [os.path.join(root, f"reference/{img_fname}") for img_fname in img_fnames
                 if img_fname.endswith((".png", ".jpg"))]
        create_path = lambda x: os.path.join(root, f"reference_magface/{x}")
        magface_emb_paths = [f"{create_path(img_fname).split('.')[0]}.pt" for img_fname in img_fnames
                             if img_fname.endswith((".png", ".jpg"))]
        self.labels = {'fr_embeds':  [torch.load(magface_emb_path).to('cpu') for magface_emb_path in magface_emb_paths]}
        self.data = ImagePaths(paths=paths, size=size, random_crop=False, labels=self.labels)


class FacesHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQTrain(size=size, keys=keys)
        d2 = FFHQTrain(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class FacesHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = CelebAHQValidation(size=size, keys=keys)
        d2 = FFHQValidation(size=size, keys=keys)
        self.data = ConcatDatasetWithIndex([d1, d2])
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex
