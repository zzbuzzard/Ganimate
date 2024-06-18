import colorsys
import random
import torchvision.transforms.functional as trf
import numpy as np
import torch
from typing import List
import pickle
import sys
from PIL import Image
from os.path import join
from util import *

from gan import GAN

root = r"C:\Users\Z\Documents\GitHub"
stylegan_code_path = join(root, "stylegan2-ada-pytorch")

download_root = join(root, "stylegan2-ada-pytorch", "models", "downloads")
ffhq_model_path = join(download_root, "2366a0cffcb890fdb0ee0a193f4e0440_https___nvlabs-fi-cdn.nvidia.com_stylegan2-ada-pytorch_pretrained_ffhq.pkl")
cat_model_path = join(download_root, "afhqcat.pkl")
human_path = join(download_root, "stylegan_human_v2_1024.pkl")


def z_from_seed(G, seed: int):
    return torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))


def random_w(G, zmul=1):
    z = torch.randn((1, G.z_dim)).to(device) * zmul
    return z_to_w(G, z)


def z_to_w(G, z):
    c = torch.zeros((1, G.c_dim)).to(device)
    return G.mapping(z.to(device), c, truncation_psi=0.5, truncation_cutoff=8)


def w_to_image(G, w):
    return G.synthesis(w, noise_mode='const', force_fp32=True)


# Convert a range -1..1 to 0..1
def to_01(img):
    return torch.clamp(img * .5 + .5, 0, 1)


class StyleGAN(GAN):
    def __init__(self, name):
        assert name in ["ffhq", "cat", "human"]
        super().__init__(f"StyleGAN-{name}", 1)

        if name == "ffhq":
            path = ffhq_model_path
            self.res = (1024, 1024)
        elif name == "cat":
            path = cat_model_path
            self.res = (512, 512)
        elif name == "human":
            path = human_path
            self.res = (1024, 512)  # height, width

        with open(path, "rb") as f:
            sys.path.append(stylegan_code_path)
            p = pickle.load(f)
            G = p["G_ema"].to(device).eval()
            if G.c_dim != 0:
                print("Expected unconditional model")

        self.G = G

    @torch.no_grad()
    def get_z(self, batch_size, trunc=0.5, z_mul=1, w_mul=1, wplus_noise=0, **kwargs) -> np.ndarray:
        # Note: 'z' is actually w+ here
        z = torch.randn((batch_size, self.G.z_dim)).to(device) * z_mul
        c = torch.zeros((batch_size, self.G.c_dim)).to(device)
        wplus = self.G.mapping(z, c, truncation_psi=trunc, truncation_cutoff=8)
        wplus *= w_mul
        wplus += torch.randn_like(wplus) * wplus_noise
        return wplus.detach().cpu().numpy()

    @torch.no_grad()
    def generate(self, z, batch_size, unclip=True, **kwargs) -> List[Image.Image]:
        N = z.shape[0]
        output = torch.zeros((N, 3, self.res[0], self.res[1]))

        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(device)

        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)

            with torch.no_grad():
                output[i: j] = self.G.synthesis(z[i: j], noise_mode='const', force_fp32=True)

        if unclip:
            output = torch.clip(output * .5 + .5, 0, 1)

        return [trf.to_pil_image(i) for i in output.unbind(0)]
