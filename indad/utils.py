from torch._C import Value
import yaml
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn, Tensor, tensor
from torchvision import transforms

import numpy as np
from PIL import ImageFilter
from sklearn import random_projection

from typing import Dict, Iterable, Callable


class GaussianBlur:
    def __init__(self, radius : int = 4):
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(
            self.unload(img[0]/map_max).filter(self.blur_kernel)
        )*map_max
        return final_map


def get_coreset_idx_randomp(
    z_lib : tensor, 
    n : int = 1000,
    eps : float = 0.90,
    float16 : bool = True,
    force_cpu : bool = False,
) -> tensor:
    """Returns n coreset idx for given z_lib.
    
    Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
    CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)

    Args:
        z_lib:      (n, d) tensor of patches.
        n:          Number of patches to select.
        eps:        Agression of the sparse random projection.
        float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
        force_cpu:  Force cpu, useful in case of GPU OOM.

    Returns:
        coreset indices
    """

    print(f"Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        print(f"DONE.                 Transformed dim = {z_lib.shape}.")
    except ValueError:
        print("Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx+1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)
    # The line below is not faster than linalg.norm, although i'm keeping it in for
    # future reference.
    # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(n-1)):
        distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) # broadcasting step
        # distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True) # broadcasting step
        min_distances = torch.minimum(distances, min_distances) # iterative step
        select_idx = torch.argmax(min_distances) # selection step

        # bookkeeping
        last_item = z_lib[select_idx:select_idx+1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    return torch.stack(coreset_idx)

def write_results(results : dict, method : str):
    """Writes results to .yaml and serialized results to .txt."""
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    name = f"{method}_{timestamp}"
    with open(f"./results/{name}.yml", "w") as outfile:
        yaml.safe_dump(results, outfile, default_flow_style=False)
    with open(f"./results/{name}.txt", "w") as outfile:
        outfile.write(serialize_results(results["per_class_results"]))

def serialize_results(results : dict) -> str:
    """Serialized a results dict into something usable in markdown."""
    n_first_col = 20
    ans = []
    for k, v in results.items():
        print(v)
        s = k + " "*(n_first_col-len(k))
        s = s + f"| {v[0]*100:.1f}  | {v[1]*100:.1f}  |"
        ans.append(s)
    return "\n".join(ans)


class FeatureExtractor(nn.Module):
    """From https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904"""
    def __init__(
        self, model: nn.Module,
        layer_names: Iterable[str],
        layer_idx_refs: Iterable[int],
    ):
        super().__init__()
        self.model = model
        self.layer_names = layer_names
        self._features = {layer: torch.empty(0) for layer in layer_idx_refs}

        for i in range(len(layer_names)):
            layer = dict([*self.model.named_modules()])[layer_names[i]]
            layer.register_forward_hook(
                self.save_outputs_hook(layer_idx_refs[i])
            )

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features
