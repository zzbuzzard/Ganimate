import numpy as np
import math
from os.path import join
import os
import shutil
from scipy.ndimage import gaussian_filter1d
import torch

import util
from gan import GAN
from pipeline import generate_from_config


def sinusoidal_walk(z, steps, seed=-1, min_cycles=1, max_cycles=3, move_class=False, amplitude=1):
    if not move_class:
        c = z[128:]
        z = z[:128]

    if seed != -1:
        np.random.seed(seed)

    sh = (len(z.shape)-1)*(1,) + (z.shape[-1],)

    bias = np.random.uniform(0, 2 * math.pi, size=sh).astype(np.float32)
    num_its = np.random.randint(min_cycles, max_cycles + 1, size=sh)

    t = np.linspace(0, 1, steps, endpoint=False, dtype=np.float32).reshape((steps,) + (len(z.shape)*(1,)))

    # t x N
    offsets = np.sin(t * 2 * math.pi * num_its[None] + bias[None]) - np.sin(bias[None])

    # bias = 1 x (shape)
    # t    = (t)

    # t x N
    zs = z[None] + offsets * amplitude

    if not move_class:
        cs = c[None].repeat(t.shape[0], 0)
        zs = np.concatenate((zs, cs), axis=1)

    return zs.astype(np.float32)


def interpolate(zs: np.ndarray, mean, std, frames_between: int, normalise: bool = False, gaussian_smooth: bool = True,
                gaussian_sigma: float = 2):
    # x, y ~ N(m, s^2)

    # (1-p)x ~ N((1-p)m, (1-p)^2 s^2)
    # py ~ N(pm, p^2 s^2)
    # (1-p)x + py ~ N((1-p)m + pm = m, [(1-p)^2 + p^2] s^2)

    # so normalise `(1-p)x + py` => subtract m, divide by ((1-p)^2+p^2)^0.5, and add m

    all_zs = []

    n = zs.shape[0]
    for i in range(n):
        start = zs[i][None]  # 1 x D
        end = zs[(i + 1) % n][None]  # 1 x D
        ps = np.linspace(0, 1, frames_between, endpoint=False, dtype=np.float32)[:, None]  # N x 1
        norm_factor = ((1 - ps) ** 2 + ps ** 2) ** 0.5
        lerped = start + ps * (end - start)  # N x D

        if normalise:
            normed = (lerped - mean) / norm_factor + mean
        else:
            normed = lerped

        all_zs.append(normed)

    all_zs = np.concatenate(all_zs, axis=0)

    if gaussian_smooth:
        all_zs = gaussian_filter1d(all_zs, sigma=gaussian_sigma, axis=0, mode='wrap')

    return all_zs


class Anim:
    """
    itemroot/
     anim_name/
      images
     anim_name.gif / .mp4
    """

    # if zs is None, loads
    def __init__(self, item_root, item_config, name, zs=None):
        self.item_root = item_root
        self.item_config = item_config
        self.name = name

        self.gif_path = join(item_root, name + ".gif")
        self.mp4_path = join(item_root, name + ".mp4")
        self.dir = join(item_root, name)
        # self.np_path = join(item_root, name + ".npy")

        self.zs = zs

        # if zs is None:
        #     self.zs = np.load(self.np_path)
        # else:
        #     self.zs = zs

    def save_images_and_make_gif(self, gan, upscaler, rembg_session, batch_size, fps=24, progress=None):
        if os.path.exists(self.gif_path):
            print("Warning: making gif but it already exists")

        imgs = generate_from_config(self.item_config, self.zs, gan, upscaler, rembg_session, batch_size, progress=progress)

        util.make_video_av(imgs, self.mp4_path, fps)

        # os.makedirs(self.dir, exist_ok=True)
        # for i, img in enumerate(imgs):
        #     img.save(join(self.dir, f"{i:06d}.png"))
        #
        # util.make_video(self.dir, fps, gif=True, out_path=self.gif_path)
        # util.make_video(self.dir, fps, gif=False, out_path=self.mp4_path)

    def save_with_name(self, name):
        item_root = self.item_root

        new_gif_path = join(item_root, name + ".gif")
        new_mp4_path = join(item_root, name + ".mp4")
        new_dir = join(item_root, name)

        shutil.move(self.gif_path, new_gif_path)
        shutil.move(self.mp4_path, new_mp4_path)
        shutil.move(self.dir, new_dir)

        self.name = name
        self.gif_path = new_gif_path
        self.mp4_path = new_mp4_path
        self.dir = new_dir

    @staticmethod
    def sin_walk(item, steps, **kwargs):
        if os.path.isdir(join(item.root, "tmp")):
            shutil.rmtree(join(item.root, "tmp"))
        if os.path.isfile(join(item.root, "tmp.gif")):
            os.remove(join(item.root, "tmp.gif"))
        if os.path.isfile(join(item.root, "tmp.mp4")):
            os.remove(join(item.root, "tmp.mp4"))

        zs = sinusoidal_walk(item.z, steps, **kwargs)
        return Anim(item.root, item.config, "tmp", zs=zs)
