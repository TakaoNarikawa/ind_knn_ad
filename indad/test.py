from re import A

from sklearn.utils import shuffle
import click
import os
import time
from torch.utils.data import DataLoader

from data import MVTecDataset, mvtec_classes, CustomDataset
from models import CACHE_DIR, SPADE, PaDiM, PatchCore
from utils import print_and_export_results
from sklearn.model_selection import train_test_split
from typing import List
import torch.multiprocessing as mp
from tqdm import tqdm
import itertools

# seeds
import torch
import threading
import glob
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

ALL_CLASSES = mvtec_classes()
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

@click.command()
@click.argument("method")
@click.option("--dataset", default="hazelnut_reduced", help="Dataset")
def cli_interface(method: str, dataset: str): 
    if dataset == "all":
        dataset = ALL_CLASSES
    else:
        dataset = [dataset]

    method = method.lower()
    assert method in ALLOWED_METHODS, f"Select from {ALLOWED_METHODS}."

    total_results = run_model(method, dataset)

    print_and_export_results(total_results, method)
    
if __name__ == "__main__":
    cli_interface()
