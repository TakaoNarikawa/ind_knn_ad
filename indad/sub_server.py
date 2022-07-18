import argparse
import glob
import os
import random
from typing import List

import numpy as np
# seeds
import sys
import json
import torch
import torch.multiprocessing as mp
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import CustomDataset
from models import CACHE_DIR, SPADE, PaDiM, PatchCore
import warnings  # for some torch warnings regarding depreciation

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--port')
parser.add_argument('--params')

args = parser.parse_args()

ALLOWED_METHODS = ["spade", "padim", "patchcore"]
DATASET_DIR = "/home/gecs/datasets/Xstamper_data"

def get_angle_patterns(cls):
    if cls == "Folder":
        return ["Front", "Side", "Top"]
    return ["Bottom", "Front", "Side"]

def get_light_patterns(cls, angle):
    if cls == "Folder" and angle == "Side":
        return [100, 180]
    if cls == "Folder":
        return [100, 200]
    return [100, 140]

def get_custom_datasets(dataset_dir: str, classes: List, x_split: int, y_split: int):
    datasets = []
    for cls in classes:
        for angle in get_angle_patterns(cls):
            for light in get_light_patterns(cls, angle):
                for split_index in range(x_split*y_split):
                    glob_path = os.path.join(dataset_dir, cls, angle, f"*/*{light}.png")
                    datasets.append(CustomDataset(glob_path=glob_path, split_index=split_index, x_split=x_split, y_split=y_split))

    return datasets

def get_train_valid_img_paths(cls, angle, light, test_size=0.1):
    glob_path = os.path.join(DATASET_DIR, cls, angle, f"ok*/*{light}.png")
    train_paths, valid_ok_paths = train_test_split(glob.glob(glob_path), test_size=test_size)
    valid_ng_paths = glob.glob(glob_path.replace("/ok*/", "/ng*/"))
    valid_paths = list(valid_ok_paths) + list(valid_ng_paths)

    return list(train_paths), list(valid_paths)

class ADM:
    '''
    照明や分割を固定して
    1つのモデルと1つのデータセットを保持する
    '''
    def __init__(self, train_paths, valid_paths=[], pattern_idx=0, split_idx=0, x_split=2, y_split=2, model_name="padim", img_size=256, adm_name='default'):
        
        assert model_name in ["spade", "padim", "patchcore"]
        self.name = adm_name
        self.split_idx = split_idx
        self.pattern_idx = pattern_idx
        
        self.cache_path = os.path.join(CACHE_DIR, f"adm_{self.name}")
        self.dataset_params = dict(split_idx=self.split_idx, x_split=x_split, y_split=y_split, img_size=img_size)
        self.train_ds = CustomDataset(train_paths, **self.dataset_params)
        self.valid_ds = CustomDataset(valid_paths, **self.dataset_params)

        self.train_dl = DataLoader(self.train_ds, num_workers=0)
        self.valid_dl = DataLoader(self.valid_ds, num_workers=0, shuffle=True)

        if model_name == "spade":
            self.model = SPADE(
                k=50,
                backbone_name="wide_resnet50_2",
            )
        elif model_name == "padim":
            self.model = PaDiM(
                d_reduced=350,
                backbone_name="wide_resnet50_2",
            )
        elif model_name == "patchcore":
            self.model = PatchCore(
                f_coreset=.10, 
                backbone_name="wide_resnet50_2",
            )

        self.model.share_memory()
    
    def fit(self):
        self.model.fit(self.train_dl)
        self.model.cache_feature_maps(path=self.cache_path)

    def restore(self):
        self.model.restore_feature_maps(path=self.cache_path)

    def evaluate(self):
        assert len(self.valid_dl) > 0
        return self.model.evaluate(self.valid_dl)

    def predict(self, img_path):
        sample, _ = self.valid_ds.getitem_base(img_path)
        samples = sample.unsqueeze(0)
        return self.model.predict(samples)

adm = ADM(**json.loads(args.params))
app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'message': 'index'})

@app.route('/fit', methods=['POST'])
def fit():
    adm.fit()
    return jsonify({'message': 'fit done'})

@app.route('/restore', methods=['POST'])
def restore():
    adm.restore()
    return jsonify({'message': 'restore done'})

@app.route('/predict', methods=['POST'])
def predict():
    img_path = request.json['img_path']
    out = adm.predict(img_path)
    return jsonify({'message': 'predict done', 'out': out})

if __name__ == '__main__':
    app.run(port=int(args.port), debug=True)
