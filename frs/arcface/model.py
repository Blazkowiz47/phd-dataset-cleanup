import numpy as np
from PIL import Image
import cv2
from numpy.typing import NDArray
import onnx
import onnxruntime

__all__ = [
    "ArcFaceONNX",
]


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = "recognition"
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            # print(nid, node.name)
            if node.name.startswith("Sub") or node.name.startswith("_minus"):
                find_sub = True
            if node.name.startswith("Mul") or node.name.startswith("_mul"):
                find_mul = True

        if find_sub and find_mul:
            # mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5

        self.input_mean = input_mean
        self.input_std = input_std
        # print('input mean and std:', self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(
                self.model_file, providers=["CUDAExecutionProvider"]
            )
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CUDAExecutionProvider"])

    def get(self, img, kps):
        #         aimg = face_align.norm_crop(img, landmark=kps, image_size=self.input_size[0])
        #         embedding = self.get_feat(aimg).flatten()
        #         return embedding
        raise NotImplementedError()

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm

        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        #         print(1.0 / self.input_std)
        #         print(input_size)
        blob = cv2.dnn.blobFromImages(
            imgs,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
        )
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out


def get_model(ckpt: str = "./models/frs_models/arcface/model.onnx"):
    return ArcFaceONNX(ckpt)


def transform(fname: str) -> NDArray:
    img = np.array(Image.open(fname).convert("RGB")).astype(np.float32) / 255.0
    return img


def get_features(fname: str, model) -> NDArray:
    img = transform(fname)
    features = model.get_feat(img)
    return features.squeeze()


if __name__ == "__main__":
    model = ArcFaceONNX(
        "/Users/sushrutpatwardhan/1Projects/frs/models/arcface/model.onnx"
    )
    imgname = "/Users/sushrutpatwardhan/1Projects/frgc/test/02463_1.jpg"
    img = np.array(Image.open(imgname).convert("RGB")).astype(np.float32) / 255.0
    features = model.get_feat(img)
    print(features.shape)
