import os
from typing import Union

import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm

import sys
sys.path.append("/mnt2/PhD-Marcel/ldm-face-manipulation/latent-diffusion/frgc_wrapper/utils")

from __init__ import BaseDataset
from file import get_file_list, get_image
from similarity import findCosineDistance
from pair_selection import make_morph_pairs



class FRGCDataset(BaseDataset):
    dataset_id = "frgc"
    # Label file: data.csv
    # images for morphing: reference/

    def __init__(self, root_path, image_set='reference', low_res_morphs=False, **kwargs):
        self.root_path = root_path
        self.image_set = image_set
        label_file = os.path.join(root_path, 'data.csv')
        _lr = 'lowres' if low_res_morphs else ''
        images_path = os.path.join(root_path, image_set, _lr)

        self.image_paths = {k: v for k, v in ((f, os.path.join(images_path, f)) for f in get_file_list(images_path))}

        if image_set == 'morph':
            _parts = [(mfn, mfn.split('_')) for mfn in self.image_paths.keys()]
            _parts = {mfn: [_p[1], _p[3]] for mfn, _p in _parts}

        # make subject id column, clean data
        labels = pd.read_csv(label_file)
        labels['S'] = labels['Name'].str.split('d', expand=True)[0]
        self.labels = labels.dropna(axis=1)

        if image_set == 'morph':
            mdf = pd.DataFrame({'Morph': _parts.keys(), 'MS': _parts.values()})
            mdf[['S1', 'S2']] = mdf['MS'].tolist()
            mdf['S1'] = mdf['S1'].str.split('d', expand=True)[0]
            mdf['S2'] = mdf['S2'].str.split('d', expand=True)[0]
            ms1 = mdf.merge(self.labels.rename(columns={'S': 'S1'}), on=['S1']).drop(columns=['Name'])
            ms1f = ms1['Morph'].duplicated()
            self.labels = ms1.loc[~ms1f].rename(columns={'Morph': 'Name'}).reset_index(drop=True)
        else:
            rev_labels = pd.DataFrame({'Name': self.image_paths.keys()})
            rev_labels['S'] = rev_labels['Name'].str.split('d', expand=True)[0]
            _subject_labels = self.labels.loc[self.labels.S.isin(rev_labels.S), ['S', 'Sex', 'Race', 'Skin Type', 'Age']].drop_duplicates()
            self.labels = _subject_labels.merge(rev_labels, on=['S'])

        self.features = {}
        self.features = self._extract_features()

    def __len__(self):
        return len(self.image_paths.keys())

    def __getitem__(self, i):
        if self.image_set == 'morph':
            fn, pair, sid1, sid2, gender, eth, skin, age = self.labels.iloc[i].to_list()
            return {
                'file': self.image_paths[fn],
                'file_pair': pair,
                'subject': (sid1, sid2),
                'gender': gender,
                'ethnicity': eth,
                'skin_type': skin,
                'age': age,
                'features': self.features.get(i, None)
            }
        else:
            sid, gender, eth, skin, age, fn = self.labels.iloc[i].to_list()
            return {
                'file': self.image_paths[fn],
                'file_pair': None,
                'subject': sid,
                'gender': gender,
                'ethnicity': eth,
                'skin_type': skin,
                'age': age,
                'features': self.features.get(i, None)
            }

    def get_subject_samples(self, subject_id: str):
        samples = self.labels.loc[self.labels.S.str.contains(subject_id)]
        return self[samples.sample(1).index[0]]
    
    def get_subject_sample_indices(self, subject_id: str):
        samples = self.labels.loc[self.labels.S.str.contains(subject_id)]
        return list(samples.index)

    def _extract_features(self):
        features = {}
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        ds = get_image(self[0]['file']).shape[0]//2
        app.prepare(ctx_id=0, det_size=(ds, ds))
        for i in tqdm(range(0, self.__len__()), desc='Extracting features'):
            # attempt loading
            feat = self._load_feature(self[i]['file'])
            if feat is not None:
                features[i] = feat
            else:
                img = get_image(self[i]['file'])
                face = next(iter(app.get(img, max_num=1)), None)
                assert face, f"Face detection error: {i} - {self[i]['file']}"
                self._save_feature(self[i]['file'], face)
                features[i] = face
        return features

    def generate_cost_matrix(self, use_similarity=True, use_labels=False):
        cmat_fn = f"{self.dataset_id}{'_sim' if use_similarity else ''}{'_lab' if use_labels else ''}{'_'+self.image_set if self.image_set != 'reference' else ''}.npy"
        cmat_fn = os.path.join(self.root_path, cmat_fn)

        if os.path.exists(cmat_fn):
            c = np.load(cmat_fn)
        else:
            c = np.zeros((len(self), len(self)), dtype=np.float32)
            for i in tqdm(range(c.shape[1]), 'Generating cost matrix'):
                row_sample = self[i]

                for j in range(c.shape[0]):
                    col_sample = self[j]

                    if row_sample['subject'] == col_sample['subject']:
                        c[i, j] = np.inf
                        continue
                    
                    # if any((True for rss in row_sample['subject'] if rss in col_sample['subject'])):
                    #     c[i, j] = np.inf
                    #     continue
                    
                    distance_bonus_factor = 0.  # no bonus

                    if use_similarity:
                        try:
                            c[i, j] = findCosineDistance(row_sample['features']['normed_embedding'], col_sample['features']['normed_embedding'])
                        except KeyError:
                            c[i, j] = findCosineDistance(row_sample['features'].normed_embedding, col_sample['features'].normed_embedding)

                    if use_labels:
                        if row_sample['gender'] == col_sample['gender']:
                            distance_bonus_factor += 0.15
                        else:
                            distance_bonus_factor -= 0.15

                        if row_sample['ethnicity'] == col_sample['ethnicity']:
                            distance_bonus_factor += 0.05
                        else:
                            distance_bonus_factor -= 0.05

                        if any(map(lambda v: v in row_sample['skin_type'].split('-'), col_sample['skin_type'].split('-'))):
                            distance_bonus_factor += 0.05

                        if any(map(lambda v: v in row_sample['age'].split('-'), col_sample['age'].split('-'))):
                            distance_bonus_factor += 0.025

                    bonus = 1. - distance_bonus_factor
                    c[i, j] *= bonus
            np.save(cmat_fn, c)

        return c