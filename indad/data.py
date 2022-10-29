import glob
import os
from random import sample
import tarfile
from configparser import Interpolation
from os.path import isdir
from pathlib import Path
from typing import Type
import numpy as np

import wget
import utils
from PIL import Image, ImageDraw
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASETS_PATH = Path("./datasets")
IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])

def mvtec_classes():
    return [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

class MVTecDataset:
    def __init__(self, cls : str, size : int = 224):
        self.cls = cls
        self.size = size
        if cls in mvtec_classes():
            self._download()
        self.train_ds = MVTecTrainDataset(cls, size)
        self.test_ds = MVTecTestDataset(cls, size)

    def _download(self):
        if not isdir(DATASETS_PATH / self.cls) and self.cls in mvtec_classes():
            print(f"   Could not find '{self.cls}' in '{DATASETS_PATH}/'. Downloading ... ")
            url = f"ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/{self.cls}.tar.xz"
            wget.download(url)
            with tarfile.open(f"{self.cls}.tar.xz") as tar:
                tar.extractall(DATASETS_PATH)
            os.remove(f"{self.cls}.tar.xz")
            print("") # force newline
        else:
            print(f"   Found '{self.cls}' in '{DATASETS_PATH}/'\n")

    def get_datasets(self):
        return self.train_ds, self.test_ds

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)

class CustomDataset(Dataset):
    def __init__(self, paths, split_idx: int, x_split: int, y_split: int, padding=0.05, img_size=256):
        super().__init__()
        
        self.split_idx = split_idx
        self.x_split = x_split
        self.y_split = y_split
        self.padding = padding

        self.img_paths = paths

        self.transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        return self.getitem_base(img_path)

    
    def getitem_base(self, img_path):
        img = self.img2tns(img_path)
        
        if "ok" in img_path:
            sample_class = 0
        else:
            sample_class = 1

        return img, sample_class

    def img2tns(self, img_src):
        if type(img_src) is str:
            img = Image.open(img_src)
        elif isinstance(img_src, Image.Image):
            img = img_src
        elif isinstance(img_src, np.ndarray):
            img = Image.fromarray(img_src)
        else:
            raise NotImplementedError()
        img = img.convert("RGB")
        img = utils.grid_split(img, idx=self.split_idx, x_split=self.x_split, y_split=self.y_split, padding=self.padding)
        img = self.transforms(img)
        return img


class MVTecTrainDataset(ImageFolder):
    def __init__(self, cls : str, size : int):
        super().__init__(
            root=DATASETS_PATH / cls / "train",
            transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        )
        self.cls = cls
        self.size = size

class MVTecTestDataset(ImageFolder):
    def __init__(self, cls : str, size : int):
        super().__init__(
            root=DATASETS_PATH / cls / "test",
            transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]),
            target_transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]),
        )
        self.cls = cls
        self.size = size
            
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        
        if "good" in path:
            target = Image.new('L', (self.size, self.size))
            sample_class = 0
        else:
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            if os.path.exists(target_path):
                target = self.loader(target_path)
            else:
                target = Image.new('L', (self.size, self.size))
                # AUC エラーを回避するために白を含める
                target_draw = ImageDraw.Draw(target)
                target_draw.rectangle([(0, 0), (self.size//2, self.size//2)], fill=255)
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target[:1], sample_class

class StreamingDataset:
    """This dataset is made specifically for the streamlit app."""
    def __init__(self, size: int = 224):
        self.size = size
        self.transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        self.samples = []
    
    def add_pil_image(self, image : Image):
        image = image.convert('RGB')
        self.samples.append(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (self.transform(sample), tensor(0.))

# test
if __name__ == "__main__":
    dataset_dir = "/home/gecs/Downloads/Xstamper_data"
    cls = "Folder"
    angle = "Front"
    light = 100
    split_idx = 3
    x_split = 2
    y_split = 2
    
    test_dataset = CustomDataset(dataset_dir, cls, angle, light, split_idx, x_split, y_split)
    #print(test_dataset.__len__())
    items = test_dataset.__getitem__(0)
    #print(items[2])
