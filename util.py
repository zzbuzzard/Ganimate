import torch
import numpy as np
import os
from os.path import join

path_saved = "items"
path_unsaved = "gens"
device = torch.device('cuda')

if not os.path.exists(path_saved):
    os.mkdir(path_saved)
if not os.path.exists(path_unsaved):
    os.mkdir(path_unsaved)


# Path w files in format 000000.png 000001.png ...
def make_video(path, fps, gif=False, out_path=None):
    if out_path is None:
        ext = "gif" if gif else "mp4"
        out_path = os.path.join(path, f"vid.{ext}")
    if gif:
        os.system(f"ffmpeg -i {path}/%06d.png {out_path}")
    else:
        os.system(f"ffmpeg -r {fps} -i {path}/%06d.png -vcodec libx264 -y {out_path} -qp 0")


def get_next_unsaved_idx():
    ds = os.listdir(path_unsaved)
    n = 0
    while f"{n}.png" in ds:
        n += 1
    return n


def get_next_item_idx():
    ds = os.listdir(path_saved)
    n = len(ds)
    while str(n) in ds:
        n += 1
    return n


def get_unsaved_item_ids():
    ds = os.listdir(path_saved)
    xs = []
    for i in ds:
        if i.endswith(".npy") and i.split(".")[0].isdecimal():
            xs.append(int(i.split(".")[0]))
    return xs


def get_saved_item_ids():
    ds = os.listdir(path_saved)
    ds = list(reversed(sorted([int(i) for i in ds if i.isdecimal()])))
    return ds


def get_anim_names(item_idx):
    root = os.path.join(path_saved, str(item_idx))
    return [i for i in os.listdir(root) if os.path.isdir(join(root, i))]
