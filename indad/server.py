from re import A

from sklearn.utils import shuffle
import click
import os
import time
import requests
from torch.utils.data import DataLoader

from data import MVTecDataset, mvtec_classes, CustomDataset
from models import CACHE_DIR, SPADE, PaDiM, PatchCore
from utils import print_and_export_results
from sklearn.model_selection import train_test_split
from typing import List
import torch.multiprocessing as mp
import signal

# seeds
import torch
import subprocess
import glob
import random
import numpy as np
import json

# res = requests.post('http://127.0.0.1:8200' + '/restore')
# print(res.text)
# exit()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

ALL_CLASSES = mvtec_classes()
ALLOWED_METHODS = ["spade", "padim", "patchcore"]
DATASET_DIR = "/home/gecs/datasets/Xstamper_data"


on_keyboard_interrupt = [lambda: exit()]

def add_on_keyboard_interrupt(new_callback):
    on_keyboard_interrupt.insert(0, new_callback)
    def handler(a,b):
        for callback in on_keyboard_interrupt:
            callback()
    signal.signal(signal.SIGINT, handler)

adm_counter = 0

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

class ADMC:
    def __init__(self, **adm_params):
        global adm_counter
        self.adm_index = adm_counter
        self.pattern_idx = adm_params["pattern_idx"]
        adm_counter += 1

        port = 8200 + self.adm_index
        self.hostname = f'http://127.0.0.1:{port}'

        proc_command = ['python3', 'indad/sub_server.py', '--port', str(port), '--params', json.dumps(adm_params)]
        stdout_file = open(os.path.join(CACHE_DIR, f'out_{port}.log'), "w")
        proc = subprocess.Popen(proc_command, stdout=stdout_file, stderr=stdout_file)

        def on_exit():
            print("admc on exit")
            proc.kill()

        add_on_keyboard_interrupt(on_exit)
    def fit(self):
        out = requests.post(self.hostname + '/fit')
        print("out", out)

    def restore(self):
        out = requests.post(self.hostname + '/restore')
        print("out", out)
    
    def predict(self, img_path):
        out = requests.post(self.hostname + '/predict', data=json.dumps({'img_path': img_path}))
        print("out", out)
        


class ADMM:
    '''
    検査対象の品種と角度を固定して
    複数のADMを保持する

    "照明パターン×分割パターン"の数のADMを持つ
    入力には照明パターン数だけの画像が入る
    '''
    def __init__(self, train_paths_list, valid_paths_list=None, x_split=2, y_split=2, model_name="spade", adm_prefix=""):
        self.x_split = x_split
        self.y_split = y_split

        valid_paths_list = valid_paths_list or [[] for _ in train_paths_list]
        assert len(train_paths_list) == len(valid_paths_list), (len(train_paths_list), len(valid_paths_list))

        self.num_patterns = len(train_paths_list)

        self.adms = [
            [
                ADMC(train_paths=train_paths, valid_paths=valid_paths, 
                    pattern_idx=i, split_idx=j, x_split=self.x_split, y_split=self.y_split, 
                    model_name=model_name, adm_name=f'{adm_prefix}{i*self.x_split*self.y_split+j}')
                for j in range(self.x_split*self.y_split)
            ]
            for i, (train_paths, valid_paths) in enumerate(zip(train_paths_list, valid_paths_list))
        ]

    def fit(self):
        self.run_each_adm(lambda adm: adm.fit())
    def restore(self):
        self.run_each_adm(lambda adm: adm.restore())

    def evaluate(self):
        scores = []
        def handler(adm):
            score = adm.evaluate()
            scores.append(score)
        self.run_each_adm(handler)
        return scores

    def run_each_adm(self, callback):
        for adm_with_splits in self.adms:
            for adm in adm_with_splits:
                callback(adm)

    def predict(self, inputs):
        def handler(adm):
            img_path = inputs[adm.pattern_idx]
            adm.predict(img_path)
        self.run_each_adm(handler)

def run_model(method: str, classes: List):
    results = {}

    for cls in classes:
        if method == "spade":
            model = SPADE(
                k=50,
                backbone_name="wide_resnet50_2",
            )
        elif method == "padim":
            model = PaDiM(
                d_reduced=350,
                backbone_name="wide_resnet50_2",
            )
        elif method == "patchcore":
            model = PatchCore(
                f_coreset=.10, 
                backbone_name="wide_resnet50_2",
            )

        print(f"\n█│ Running {method} on {cls} dataset.")
        print(  f" ╰{'─'*(len(method)+len(cls)+23)}\n")
        train_ds, test_ds = MVTecDataset(cls).get_dataloaders()

        print("   Training ...")
        model.fit(train_ds)
        print("   Testing ...")
        image_rocauc, pixel_rocauc = model.evaluate(test_ds)
        
        print(f"\n   ╭{'─'*(len(cls)+15)}┬{'─'*20}┬{'─'*20}╮")
        print(  f"   │ Test results {cls} │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │")
        print(  f"   ╰{'─'*(len(cls)+15)}┴{'─'*20}┴{'─'*20}╯")
        results[cls] = [float(image_rocauc), float(pixel_rocauc)]
        
    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results)/len(image_results)
    image_results = [v[1] for _, v in results.items()]
    average_pixel_roc_auc = sum(image_results)/len(image_results)

    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results
    
if __name__ == "__main__":
    #cli_interface()
    
    # ADM作成
    if False:
        train_paths, valid_paths = get_train_valid_img_paths("Folder", "Side", 100)
        adm = ADM(train_paths=train_paths, valid_paths=valid_paths, split_idx=0, x_split=2, y_split=2, model_name="patchcore")
        adm.fit()
        adm.evaluate()

    # ADMM作成
    if True:
        cls, angle = "Folder", "Side"
        train_valid_paths_list = [get_train_valid_img_paths(cls, angle, light) for light in get_light_patterns(cls, angle)][:1]
        # train_valid_paths_list = train_valid_paths_list # 数を増やしてデバッグ
        train_paths_list = [p[0] for p in train_valid_paths_list]
        valid_paths_list = [p[1] for p in train_valid_paths_list]
        admm = ADMM(train_paths_list, valid_paths_list=valid_paths_list, model_name="patchcore", adm_prefix=f"{cls}_{angle}_")
        
        # subserver の起動を待つ
        time.sleep(20)
        
        # admm.fit()
        admm.restore()

        valid_img_paths = [v[0] for v in valid_paths_list]

        elapsed_times = []
        all_start = time.time()

        for _ in range(1):
            start = time.time()
            admm.predict(valid_img_paths)
            elapsed_times.append(time.time() - start)
        elapsed_times = np.array(elapsed_times)
        print (f"[elapsed_time] mean:{elapsed_times.mean()}, std:{elapsed_times.std()} "
               + f"max:{elapsed_times.max()}, min:{elapsed_times.min()} all:{time.time() - all_start}")
        exit()
        # scores = admm.evaluate()
        # scores = np.array(scores)
        # print (f"[auc] mean:{scores.mean()}, "
        #        + f"max:{scores.max()}, min:{scores.min()}")


        