import os

import numpy as np
from numpy.linalg import norm as l2norm


class BaseDataset():
    def __init__(self, root_path, **kwargs):
        self.root_path = root_path

    def _load_feature(self, image_fn: str):
        features_path = os.path.join(self.root_path, 'features')
        feat_base = os.path.splitext(os.path.basename(image_fn))[0]
        if 'lowres' in image_fn:
            feat_base += '_lr'
        fn = os.path.join(features_path, f"{feat_base}.npy")

        if not os.path.exists(fn):
            return None
        else:
            _feat = np.load(fn, allow_pickle=True)
            if _feat is None:
                return None

            feat = {}
            for i, k in enumerate(_feat[0]):
                feat[k] = _feat[1][i]
            feat['normed_embedding'] = self.normed_embedding(feat['embedding'])
        return feat

    def _save_feature(self, image_fn: str, feature):
        features_path = os.path.join(self.root_path, 'features')
        os.makedirs(features_path, exist_ok=True)
        feat_base = os.path.splitext(os.path.basename(image_fn))[0]
        if 'lowres' in image_fn:
            feat_base += '_lr'

        fn = os.path.join(features_path, f"{feat_base}.npy")

        feat_list = []
        fkeys = []
        for k in feature.keys():
            fkeys.append(k)
            feat_list.append(feature[k])
        np.save(fn, np.array([fkeys, feat_list], dtype=object), allow_pickle=True)

    @staticmethod
    def normed_embedding(embedding):
        if embedding is None:
            return None
        embedding_norm = l2norm(embedding)
        return embedding / embedding_norm

    @staticmethod
    def sex(gender):
        if gender is None:
            return None
        return 'male' if gender == 1 else 'female'
