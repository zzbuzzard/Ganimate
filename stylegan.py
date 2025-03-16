import colorsys
import os.path
import random
import torchvision.transforms.functional as trf
import numpy as np
import torch
from typing import List
import pickle
import sys
from PIL import Image
from os.path import join
from tqdm import tqdm
import requests
from util import *

from gan import GAN

stylegan_code_path = os.path.abspath("stylegan3")

model_root = "models/stylegan"
os.makedirs(model_root, exist_ok=True)

v2_download_root = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/"
v3_download_root = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"
all_models = ['stylegan3-t-ffhq-1024x1024', 'stylegan3-t-ffhqu-1024x1024', 'stylegan3-t-ffhqu-256x256', 'stylegan3-r-ffhq-1024x1024', 'stylegan3-r-ffhqu-1024x1024', 'stylegan3-r-ffhqu-256x256', 'stylegan3-t-metfaces-1024x1024', 'stylegan3-t-metfacesu-1024x1024', 'stylegan3-r-metfaces-1024x1024', 'stylegan3-r-metfacesu-1024x1024', 'stylegan3-t-afhqv2-512x512', 'stylegan3-r-afhqv2-512x512', 'stylegan2-ffhq-1024x1024', 'stylegan2-ffhq-512x512', 'stylegan2-ffhq-256x256', 'stylegan2-ffhqu-1024x1024', 'stylegan2-ffhqu-256x256', 'stylegan2-metfaces-1024x1024', 'stylegan2-metfacesu-1024x1024', 'stylegan2-afhqv2-512x512', 'stylegan2-afhqcat-512x512', 'stylegan2-afhqdog-512x512', 'stylegan2-afhqwild-512x512', 'stylegan2-brecahad-512x512', 'stylegan2-cifar10-32x32', 'stylegan2-celebahq-256x256', 'stylegan2-lsundog-256x256']
def parse_model_name(full_name):
    vs = full_name.split("-")

    save_path = join(model_root, full_name + ".pkl")

    res_x = vs[-1]
    height, width = map(int, res_x.split("x"))

    assert vs[0] in ["stylegan2", "stylegan3"]

    if vs[0] == "stylegan2":
        download_url = v2_download_root + full_name + ".pkl"
    else:
        download_url = v3_download_root + full_name + ".pkl"

    return (height, width), save_path, download_url

# ffhq_model_path = join(download_root, "2366a0cffcb890fdb0ee0a193f4e0440_https___nvlabs-fi-cdn.nvidia.com_stylegan2-ada-pytorch_pretrained_ffhq.pkl")
# cat_model_path = join(download_root, "afhqcat.pkl")
# human_path = join(download_root, "stylegan_human_v2_1024.pkl")


def z_from_seed(G, seed: int):
    return torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))


def random_w(G, zmul=1, n=1):
    z = torch.randn((n, G.z_dim)).to(device) * zmul
    return z_to_w(G, z)


def z_to_w(G, z):
    c = torch.zeros((z.shape[0], G.c_dim)).to(device)
    return G.mapping(z.to(device), c, truncation_psi=0.5, truncation_cutoff=8)


def w_to_image(G, w):
    return G.synthesis(w, noise_mode='const', force_fp32=True)


# Convert a range -1..1 to 0..1
def to_01(img):
    return torch.clamp(img * .5 + .5, 0, 1)


class StyleGAN(GAN):
    def __init__(self, name):
        name = name.lower()
        assert name in all_models, f"Unknown StyleGAN model '{name}'"
        res, save_path, download_url = parse_model_name(name)
        super().__init__(name, 1, res)

        if not os.path.isfile(save_path):
            print("Could not find model at", save_path)
            print("Trying to download from", download_url)

            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))  # Get file size (if available)
                with open(save_path, "wb") as f, tqdm(
                        total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as progress:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress.update(len(chunk))  # Update progress bar

            print("Done!")

        print("Loading StyleGAN model", name, "from", save_path)

        with open(save_path, "rb") as f:
            # don't blame me for stylegan's cursed distribution as pkl files... safetensors wasn't around yet!
            sys.path.append(stylegan_code_path)
            p = pickle.load(f)
            G = p["G_ema"].to(device).eval()
            if G.c_dim != 0:
                print("Expected unconditional model")

        self.G = G

        # Approximate distribution of the W space as N(μ, σ)
        ws = random_w(G, zmul=1, n=2048)[:, 0]  # n x 512
        mean_latent = torch.mean(ws, dim=0).to(device)
        centred = ws - mean_latent

        self.mean_latent = mean_latent.cpu().detach().numpy()
        self.std_latent = torch.std(centred, dim=0).cpu().detach().numpy()

        self.mean_latent_tensor = mean_latent

    @torch.no_grad()
    def get_z(self, batch_size, trunc=0.5, z_mul=1, w_mul=1, wplus_noise=0, **kwargs) -> np.ndarray:
        # Note: 'z' is actually w+ here
        z = torch.randn((batch_size, self.G.z_dim)).to(device) * z_mul
        c = torch.zeros((batch_size, self.G.c_dim)).to(device)
        wplus = self.G.mapping(z, c, truncation_psi=trunc, truncation_cutoff=8)

        wplus = (wplus - self.mean_latent_tensor) * w_mul + self.mean_latent_tensor
        # wplus *= w_mul

        wplus += torch.randn_like(wplus) * wplus_noise
        return wplus.detach().cpu().numpy()

    @torch.no_grad()
    def generate(self, z, batch_size, progress=None, unclip=True, **kwargs) -> List[Image.Image]:
        N = z.shape[0]
        output = torch.zeros((N, 3, self.res[0], self.res[1]))

        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).to(device)

        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)

            with torch.no_grad():
                output[i: j] = self.G.synthesis(z[i: j], noise_mode='const', force_fp32=True)

            if progress is not None:
                progress((j, N), desc="Generating images...")

        if unclip:
            output = torch.clip(output * .5 + .5, 0, 1)

        return [trf.to_pil_image(i) for i in output.unbind(0)]

