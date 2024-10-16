import os
import numpy as np
import albumentations
import torch
import random
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

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
        root = "data/ffhq"
        with open("data/ffhqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


class FFHQValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/ffhq"
        with open("data/ffhqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.keys = keys


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
    
class AgeDBTraining(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/home/hda10308/data/imt6171"
        img_fnames = os.listdir(os.path.join(root, "TrainingDB-aligned"))
        magface_embed_fnames = os.listdir(os.path.join(root, "TrainingDB-magface"))
        mivolo_embed_fnames = os.listdir(os.path.join(root, "TrainingDB-mivolo"))
        img_paths = [os.path.join(root, f"TrainingDB-aligned/{img_fname}") for img_fname in img_fnames
                 if img_fname.endswith(".png")]
        magface_embed_paths = [os.path.join(root, f"TrainingDB-magface/{magface_embed_fname}") for magface_embed_fname in magface_embed_fnames
                 if magface_embed_fname.endswith(".pt")]
        mivolo_embed_paths = [os.path.join(root, f"TrainingDB-mivolo/{mivolo_embed_fname}") for mivolo_embed_fname in mivolo_embed_fnames
                 if mivolo_embed_fname.endswith(".pt")]
        
        concated_embeds = []
        mivolo_embeds = []
        for i in range(len(magface_embed_paths)):
            magface_embed = torch.load(magface_embed_paths[i]).to('cpu')
            mivolo_embed = torch.load(mivolo_embed_paths[i]).to('cpu')[0].unsqueeze(0)
            concated_embed = torch.cat((magface_embed, mivolo_embed), dim=1)
            #concated_embeds.append(concated_embed)
            mivolo_embeds.append(mivolo_embed)
            
        self.labels = {'concated_embeds':  mivolo_embeds}
        self.data = ImagePaths(paths=img_paths, size=size, random_crop=False, labels=self.labels)
        
class AgeDBValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/home/hda10308/data/imt6171"
        img_fnames = os.listdir(os.path.join(root, "ValidationDB-aligned"))
        magface_embed_fnames = os.listdir(os.path.join(root, "ValidationDB-magface"))
        mivolo_embed_fnames = os.listdir(os.path.join(root, "ValidationDB-mivolo"))
        img_paths = sorted([os.path.join(root, f"ValidationDB-aligned/{img_fname}") for img_fname in img_fnames
                 if img_fname.endswith(".png")])
        magface_embed_paths = sorted([os.path.join(root, f"ValidationDB-magface/{magface_embed_fname}") for magface_embed_fname in magface_embed_fnames
                 if magface_embed_fname.endswith(".pt")])
        mivolo_embed_paths = sorted([os.path.join(root, f"ValidationDB-mivolo/{mivolo_embed_fname}") for mivolo_embed_fname in mivolo_embed_fnames
                 if mivolo_embed_fname.endswith(".pt")])
        
        concated_embeds = []
        mivolo_embeds = []
        for i in range(len(magface_embed_paths)):
            magface_embed = torch.load(magface_embed_paths[i]).to('cpu')
            mivolo_embed = torch.load(mivolo_embed_paths[i]).to('cpu')[0].unsqueeze(0)
            concated_embed = torch.cat((magface_embed, mivolo_embed), dim=1)
            #concated_embeds.append(concated_embed)
            mivolo_embeds.append(mivolo_embed)
            
        self.labels = {'concated_embeds':  mivolo_embeds}
        self.data = ImagePaths(paths=img_paths, size=size, random_crop=False, labels=self.labels)
        

class AgeDBClipTraining(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/home/hda10308/data/imt6171"
        img_fnames = os.listdir(os.path.join(root, "TrainingDB-aligned"))
        magface_embed_fnames = os.listdir(os.path.join(root, "TrainingDB-magface"))
        img_paths = sorted([os.path.join(root, f"TrainingDB-aligned/{img_fname}") for img_fname in img_fnames
                 if img_fname.endswith(".png")])
        magface_embed_paths = sorted([os.path.join(root, f"TrainingDB-magface/{magface_embed_fname}") for magface_embed_fname in magface_embed_fnames
                 if magface_embed_fname.endswith(".pt")])
        
        clip_age_cond_strs = []
        magface_embeds = []
        for i in range(len(magface_embed_paths)):
            # create CLIP age conditioning string
            age = int(img_paths[i].split("_")[-1].split(".")[0])
            if 0 <= age < 4:
                age_group = random.choice(["baby", "infant", "newborn"])
            elif 4 <= age < 13:
                age_group =  random.choice(["child", "kid"])
            elif 13 <= age < 20:
                age_group = random.choice(["teenager", "adolescent", "teen"])
            elif 20 <= age < 30:
                age_group = "young adult"
            elif 30 <= age < 60:
                age_group = "adult"
            elif 60 <= age < 80:
                age_group = "senior"
            elif 80 <= age:
                age_group = "elder"
            clip_age_cond_str = f"A {age} year old {age_group}"
            clip_age_cond_strs.append(clip_age_cond_str)
            
            # create id conditioning with MagFace embedding
            magface_embed = torch.load(magface_embed_paths[i]).to('cpu').squeeze()
            magface_embeds.append(magface_embed)
            
        self.labels = {'conds':  {'clip_age_cond_strs': clip_age_cond_strs,
                                  'magface_embeds': magface_embeds}}
        self.data = ImagePaths(paths=img_paths, size=size, random_crop=False, labels=self.labels)
        
        
class AgeDBClipValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/home/hda10308/data/imt6171"
        img_fnames = os.listdir(os.path.join(root, "ValidationDB-aligned"))
        magface_embed_fnames = os.listdir(os.path.join(root, "ValidationDB-magface"))
        img_paths = sorted([os.path.join(root, f"ValidationDB-aligned/{img_fname}") for img_fname in img_fnames
                 if img_fname.endswith(".png")])
        magface_embed_paths = sorted([os.path.join(root, f"ValidationDB-magface/{magface_embed_fname}") for magface_embed_fname in magface_embed_fnames
                 if magface_embed_fname.endswith(".pt")])

        
        clip_age_cond_strs = []
        magface_embeds = []
        for i in range(len(magface_embed_paths)):
            # create CLIP age conditioning string
            age = int(img_paths[i].split("_")[-1].split(".")[0])
            if 0 <= age < 4:
                age_group = "baby"
            elif 4 <= age < 13:
                age_group =  "child"
            elif 13 <= age < 20:
                age_group = "teenager"
            elif 20 <= age < 30:
                age_group = "young adult"
            elif 30 <= age < 60:
                age_group = "adult"
            elif 60 <= age < 80:
                age_group = "senior"
            elif 80 <= age:
                age_group = "elder"
            clip_age_cond_str = f"A {age} year old {age_group}"
            clip_age_cond_strs.append(clip_age_cond_str)
            
            # create id conditioning with MagFace embedding
            magface_embed = torch.load(magface_embed_paths[i]).to('cpu').squeeze()
            magface_embeds.append(magface_embed)
            
        self.labels = {'conds':  {'clip_age_cond_strs': clip_age_cond_strs,
                                  'magface_embeds': magface_embeds}}
        self.data = ImagePaths(paths=img_paths, size=size, random_crop=False, labels=self.labels)
        

class AgeDBClipDev(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "/mnt2/datasets/face/Marcel-CustomDBs/imt6171"
        img_fnames = os.listdir(os.path.join(root, "DevDB-aligned"))
        magface_embed_fnames = os.listdir(os.path.join(root, "DevDB-magface"))
        img_paths = sorted([os.path.join(root, f"DevDB-aligned/{img_fname}") for img_fname in img_fnames
                 if img_fname.endswith(".png")])
        magface_embed_paths = sorted([os.path.join(root, f"DevDB-magface/{magface_embed_fname}") for magface_embed_fname in magface_embed_fnames
                 if magface_embed_fname.endswith(".pt")])
        
        clip_age_cond_strs = []
        magface_embeds = []
        for i in range(len(magface_embed_paths)):
            # create CLIP age conditioning string
            age = int(img_paths[i].split("_")[-1].split(".")[0])
            if 0 <= age < 4:
                age_group = "baby"
            elif 4 <= age < 13:
                age_group =  "child"
            elif 13 <= age < 20:
                age_group = "teenager"
            elif 20 <= age < 30:
                age_group = "young adult"
            elif 30 <= age < 60:
                age_group = "adult"
            elif 60 <= age < 80:
                age_group = "senior"
            elif 80 <= age:
                age_group = "elder"
            clip_age_cond_str = f"A {age} year old {age_group}"
            clip_age_cond_strs.append(clip_age_cond_str)
            
            # create id conditioning with MagFace embedding
            magface_embed = torch.load(magface_embed_paths[i])#.squeeze()
            magface_embeds.append(magface_embed)
            
        self.labels = {'conds':  {'clip_age_cond_strs': clip_age_cond_strs,
                                  'magface_embeds': magface_embeds}}
        self.data = ImagePaths(paths=img_paths, size=size, random_crop=False, labels=self.labels)
