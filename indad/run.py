import hashlib
import os
from torch.utils.data import DataLoader

from data import  CustomDataset
from models import CACHE_DIR, SPADE, PaDiM, PatchCore
from sklearn.model_selection import train_test_split
from typing import List
import itertools

# seeds
import torch
from PIL import Image
import glob
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

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
                    glob_path = os.path.join(dataset_dir, cls, angle, "*", f"*{light}.png")
                    datasets.append(CustomDataset(glob_path=glob_path, split_index=split_index, x_split=x_split, y_split=y_split))

    return datasets

def get_train_valid_img_paths(cls, angle, light, test_size=0.1):
    glob_path = os.path.join(DATASET_DIR, cls, angle, "ok*", f"*{light}.png")
    train_paths, valid_ok_paths = train_test_split(glob.glob(glob_path), test_size=test_size)
    valid_ng_paths = glob.glob(
        glob_path.replace("/ok*/", "/ng*/")
                 .replace("\\ok*\\", "\\ng*\\") # windows 用
    )
    valid_paths = list(valid_ok_paths) + list(valid_ng_paths)
    
    # string[], string[]
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
        self.valid_dl = DataLoader(self.valid_ds, num_workers=0, shuffle=(len(self.valid_ds) > 0))

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
        self.model.set_threshold(self.train_dl, self.valid_dl, determine_type=0)
        self.model.cache_feature_maps(path=self.cache_path)

    def restore(self):
        self.model.restore_feature_maps(path=self.cache_path)

    def evaluate(self):
        if not len(self.valid_dl) > 0:
            print("検証用データが存在しないのでテストできませんでした")
            return 0
        return self.model.evaluate(self.valid_dl)

    def predict(self, sample):
        samples = sample.unsqueeze(0)
        return self.model.predict(samples)


class ADMM:
    '''
    検査対象の品種と角度を固定して
    複数のADMを保持する

    "照明パターン×分割パターン"の数のADMを持つ
    入力には照明パターン数だけの画像が入る
    '''
    def __init__(self, train_paths_list, valid_paths_list=None, x_split=1, y_split=1, img_size=256, model_name="spade", adm_prefix=""):
        self.x_split = x_split
        self.y_split = y_split

        valid_paths_list = valid_paths_list or [[] for _ in train_paths_list]
        assert len(train_paths_list) == len(valid_paths_list), (len(train_paths_list), len(valid_paths_list))

        self.num_patterns = len(train_paths_list)

        self.adms = [
            [
                ADM(train_paths=train_paths, valid_paths=valid_paths, 
                    pattern_idx=i, split_idx=j, x_split=self.x_split, y_split=self.y_split, img_size=img_size,
                    model_name=model_name, adm_name=f'{adm_prefix}{i*self.x_split*self.y_split+j}')
                for j in range(self.x_split*self.y_split)
            ]
            for i, (train_paths, valid_paths) in enumerate(zip(train_paths_list, valid_paths_list))
        ]

    def fit(self):
        self.run_each_adm(lambda adm: adm.fit())
    def restore(self):
        self.run_each_adm(lambda adm: adm.restore())
    def set_threshold(self):
        self.run_each_adm(lambda adm: adm.set_threshold())

    def evaluate(self):
        scores = []
        def handler(adm):
            score = adm.evaluate()
            scores.append(score)
        self.run_each_adm(handler)
        return scores

    def run_each_adm(self, callback):
        for adm in itertools.chain.from_iterable(self.adms):
            callback(adm)

    def predict(self, inputs):
        preds = []
        pred_labels = []
        def handler(adm):
            img_src = inputs[adm.pattern_idx]
            img_tns = adm.valid_ds.img2tns(img_src)
            out, _ = adm.predict(img_tns)

            pred = (out < adm.model.threshold).item() if adm.model.threshold is not None else None
            pred_label = ('OK' if pred else 'NG') if pred is not None else None
            
            print('-'*5)
            print(f"score: {out.item():.3} -> {pred_label}")

            if type(img_src) is str:
                label = 0 if "ok" in img_src else 1
                print(f"label: {['OK', 'NG'][label]}")
            
            preds.append(pred)
            pred_labels.append(pred_label)

        self.run_each_adm(handler)
        return preds, pred_labels
    
    @classmethod
    def from_dirpath(cls, dirpath, evaluate=False, **kwargs):
        adm_dirs = [os.path.join(dirpath, p) for p in os.listdir(dirpath)]

        adm_dirs = [{'ok': os.path.join(d, 'ok'), 'ng': os.path.join(d, 'ng')} 
                        for d in adm_dirs 
                        if os.path.isdir(d) and os.path.isdir(os.path.join(d, 'ok'))] # ok ディレクトリは必須
        adm_dirs.sort(key=lambda x: x['ok'])

        print("-"*5)
        print("対象のディレクトリ一覧")
        print("推定の際にはこの順番で入力画像を与えてください")
        print('\n'.join([d['ok'] for d in adm_dirs]))
        print("-"*5)

        adm_dirs = [
            {'ok': d['ok'], 
             'ng': os.path.isdir(d['ng']) and d['ng'] or None }
                for d in adm_dirs]
        
        # string[][]
        paths_list = [{'ok': glob.glob(os.path.join(d['ok'], '*.png')), 
                       'ng': glob.glob(os.path.join(d['ng'], '*.png')) if d['ng'] else [] } 
                            for d in adm_dirs]
        def make_train_valid_paths(ok_paths, ng_paths):
            if evaluate:
                return ok_paths, []

            ok_train, ok_valid = train_test_split(ok_paths, test_size=0.5)
            train_paths = ok_train
            valid_paths = ok_valid + ng_paths
            return train_paths, valid_paths

        # [string[], string[]][]
        train_valid_paths_list = [make_train_valid_paths(d['ok'], d['ng']) for d in paths_list]
        train_paths_list = [d[0] for d in train_valid_paths_list]
        valid_paths_list = [d[1] for d in train_valid_paths_list]

        # ファイル名に使用されるので、"/"を除去するために MD5 でハッシュ化する
        adm_prefix = hashlib.md5(dirpath.encode()).hexdigest()
        
        admm = cls(train_paths_list=train_paths_list, valid_paths_list=valid_paths_list, adm_prefix=adm_prefix, **kwargs)
        admm.fit()
        if evaluate:
            admm.evaluate()
        return admm

if __name__ == "__main__":
    # ADMM で使用するディレクトリ名を与えます
    admm = ADMM.from_dirpath("./datasets/dataset_2022_10_28/holder/A/front", 
                model_name="patchcore", x_split=1, y_split=1, img_size=128, evaluate=False)
    # 入力は 画像パス, PillowImage, Numpy配列 のいずれかで与えてください
    inputs = [
        "./datasets/dataset_2022_10_28/holder/A/front/00_LED1/ng/000.png",
        Image.open("./datasets/dataset_2022_10_28/holder/A/front/01_LED2/ok/000.png"),
        np.array(Image.open("./datasets/dataset_2022_10_28/holder/A/front/02_LED3/ng/000.png"))
    ]

    preds, pred_labels = admm.predict(inputs)
    print(preds) # [False, True, False]
    print(pred_labels) # ['NG', 'OK', 'NG']

    result = all(preds)
    print(result) # False