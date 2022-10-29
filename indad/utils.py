import sys
import yaml
from tqdm import tqdm
from datetime import datetime

import torch
import numpy as np
from torch import tensor
from torchvision import transforms

from PIL import Image, ImageFilter
from scipy.stats import norm
import io
import matplotlib.pyplot as plt
from sklearn import random_projection

TQDM_PARAMS = {
	"file" : sys.stdout,
	"bar_format" : "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
}

def get_tqdm_params():
    return TQDM_PARAMS

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

    print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
    except ValueError:
        print( "   Error: could not project vectors. Please increase `eps`.")

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

    for _ in tqdm(range(n-1), **TQDM_PARAMS):
        distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) # broadcasting step
        # distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True) # broadcasting step
        min_distances = torch.minimum(distances, min_distances) # iterative step
        select_idx = torch.argmax(min_distances) # selection step

        # bookkeeping
        last_item = z_lib[select_idx:select_idx+1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    return torch.stack(coreset_idx)

def print_and_export_results(results : dict, method : str):
    """Writes results to .yaml and serialized results to .txt."""
    
    print("\n   ╭────────────────────────────╮")
    print(  "   │      Results summary       │")
    print(  "   ┢━━━━━━━━━━━━━━━━━━━━━━━━━━━━┪")
    print( f"   ┃ average image rocauc: {results['average image rocauc']:.2f} ┃")
    print( f"   ┃ average pixel rocauc: {results['average pixel rocauc']:.2f} ┃")
    print(  "   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

    # write
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    name = f"{method}_{timestamp}"

    results_yaml_path = f"./results/{name}.yml"
    scoreboard_path = f"./results/{name}.txt"

    with open(results_yaml_path, "w") as outfile:
        yaml.safe_dump(results, outfile, default_flow_style=False)
    with open(scoreboard_path, "w") as outfile:
        outfile.write(serialize_results(results["per_class_results"]))
        
    print(f"   Results written to {results_yaml_path}")

def serialize_results(results : dict) -> str:
    """Serialize a results dict into something usable in markdown."""
    n_first_col = 20
    ans = []
    for k, v in results.items():
        s = k + " "*(n_first_col-len(k))
        s = s + f"| {v[0]*100:.1f}  | {v[1]*100:.1f}  |"
        ans.append(s)
    return "\n".join(ans)

def grid_split(image, idx=0, x_split=2, y_split=2, padding=0.05):
    ix, iy = idx % x_split, idx // x_split

    x_block_size = (1 - padding * (x_split - 1)) / x_split
    y_block_size = (1 - padding * (y_split - 1)) / y_split

    xmin = x_block_size * ix
    xmax = x_block_size * (ix + 1) + padding
    ymin = y_block_size * iy
    ymax = y_block_size * (iy + 1) + padding

    w, h = image.size
    xmin, xmax = int(xmin * w), int(xmax * w)
    ymin, ymax = int(ymin * h), int(ymax * h)
    return image.crop((xmin, ymin, xmax, ymax))

def norm_ppf(mu, var, q=0.95):
    return norm.ppf(q=q, loc=mu, scale=var**0.5)

def norm_cdf(mu, var, x):
    return norm.cdf(x=x, loc=mu, scale=var)


def fmap_to_img(img, fmap, value_range=None):
    if value_range is None:
        value_range = (fmap.min(), fmap.max())
    fmap = pred_to_img(fmap, value_range)
    plt.imshow(np.array(img))
    plt.imshow(fmap, cmap="jet", alpha=0.5)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    return Image.open(buf)

def pred_to_img(x, value_range):
    range_min, range_max = value_range
    x -= range_min
    if (range_max - range_min) > 0:
        x /= (range_max - range_min)
    return tensor_to_img(x)
def tensor_to_img(x, normalize=False):
    if normalize:
        x *= tensor([.229, .224, .225]).unsqueeze(-1).unsqueeze(-1)
        x += tensor([.485, .456, .406]).unsqueeze(-1).unsqueeze(-1)
    x =  x.clip(0.,1.).permute(1,2,0).detach().numpy()
    return x