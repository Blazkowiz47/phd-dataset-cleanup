import cv2
import os
import numpy as np
from tqdm import tqdm
import sys
import os
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
import mxnet as mx
os.environ["MXNET_SUBGRAPH_VERBOSE"] = '0'
from network_inf import builder_inf
from mtcnn_detector import MtcnnDetector
from torchvision import transforms
import face_preprocess
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from easydict import EasyDict


#----------------------------------------------------------------------------
def detect_face(img, det_model):
    reshaped_img = np.reshape(img, (int(img.shape[0]), int(img.shape[1]), 3))
    bbox, points = det_model.detect_face(reshaped_img)
    single_face_bbox = bbox[0, 0:4]
    single_face_points = points[0,:].reshape((2,5)).T
    aligend_face_img = face_preprocess.preprocess(img, single_face_bbox, single_face_points, image_size='112,112')
    return aligend_face_img
#----------------------------------------------------------------------------
def load_face_detection_model(det_model_path):
    model_path = os.path.join(det_model_path)
    det_model = MtcnnDetector(model_folder=model_path, ctx=mx.cpu(), num_worker=1, accurate_landmark = True, threshold=[0.1,0.1,0.1])
    return(det_model)
#----------------------------------------------------------------------------
def load_face_recognition_model(fr_model_path):
    args = EasyDict({'arch': 'iresnet100', 'embedding_size': 512, 'resume': fr_model_path,
                'cpu_mode': False})
    fr_model = builder_inf(args)
    fr_model.eval()
    return(fr_model)
#----------------------------------------------------------------------------
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0., 0., 0.],
        std=[1., 1., 1.]),
])
#----------------------------------------------------------------------------

if __name__ == "__main__":
    # remember to move your images to the .../reference folder
    img_path = os.path.join(parent_dir, 'taming-transformers/taming/data/morph_db/reference')
    img_fnames = os.listdir(img_path)
    fr_model_path = os.path.join(parent_dir, 'models/magface/magface_epoch_00025.pth')
    det_model_path = os.path.join(parent_dir, 'models/magface/mtcnn-model')
    outdir = os.path.join(parent_dir, 'taming-transformers/taming/data/morph_db/reference_magface')

    # load magface recognition model
    detection_model= load_face_detection_model(det_model_path=det_model_path)
    fr_model= load_face_recognition_model(fr_model_path=fr_model_path)

    # compute face embeddings and save to target file
    for img_fname in tqdm(img_fnames):
        full_img_path = os.path.join(img_path, img_fname)
        target_pt_path = os.path.join(outdir, img_fname[0:-4]) + ".pt"
        if os.path.isfile(f"{target_pt_path}"): # skip if embedding already exists
            continue
        img = cv2.imread(full_img_path)
        try:
            detected_face = detect_face(img, detection_model)
        except:
            continue
        detected_face = torch.reshape(trans(detected_face), (1, 3, 112, 112))
        face_embedding = fr_model(detected_face)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        face_embedding = face_embedding.data.cpu()
        torch.save(face_embedding, target_pt_path)









