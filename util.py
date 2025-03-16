import torch
import numpy as np
import os
from os.path import join
from PIL import Image
from typing import List
import av
import torchvision.transforms.functional as trf

path_saved = "items"
path_unsaved = "gens"
device = torch.device('cuda')

if not os.path.exists(path_saved):
    os.mkdir(path_saved)
if not os.path.exists(path_unsaved):
    os.mkdir(path_unsaved)


_orig = torch.nn.Conv2d.__init__
def set_tile_mode(tiles=False):
    if tiles == get_tile_mode():
        return
    if tiles:
        def __init__(self, *args, **kwargs):
            return _orig(self, *args, **kwargs, padding_mode='circular')
        torch.nn.Conv2d.__init__ = __init__
        print("Tiled generation ENABLED!")
    else:
        torch.nn.Conv2d.__init__ = _orig
        print("Tiled generation DISABLED!")

def get_tile_mode():
    return torch.nn.Conv2d.__init__ != _orig

# Path w files in format 000000.png 000001.png ...
def make_video(path, fps, gif=False, out_path=None):
    if out_path is None:
        ext = "gif" if gif else "mp4"
        out_path = os.path.join(path, f"vid.{ext}")
    if gif:
        os.system(f"ffmpeg -i {path}/%06d.png {out_path}")
    else:
        os.system(f"ffmpeg -r {fps} -i {path}/%06d.png -vcodec libx264 -y {out_path} -qp 0")


# Uses PyAV and writes as MP4
class AvVideoWriter:
    def __init__(self, height, width, out_path, fps, crf=20):
        self.file = open(out_path, "wb")
        self.output = av.open(self.file, 'w', format="mp4")
        self.stream = self.output.add_stream('h264', str(fps))
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = 'yuv444p'
        self.stream.options = {'crf': str(crf)}  # higher value = more compression

    def write(self, img):
        if isinstance(img, Image.Image):
            arr = np.array(img)
        elif isinstance(img, np.ndarray):
            arr = img
        elif isinstance(img, torch.Tensor):
            arr = np.array(trf.to_pil_image(img))

        frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
        packet = self.stream.encode(frame)
        self.output.mux(packet)

    def close(self):
        packet = self.stream.encode(None)
        self.output.mux(packet)
        self.output.close()
        self.file.close()


def make_video_av(images: List[Image.Image], out_path, fps, crf=20):
    writer = AvVideoWriter(images[0].height, images[0].width, out_path, fps, crf=crf)
    for img in images:
        writer.write(img)
    writer.close()



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
    print(os.listdir(root))
    out = sorted(set([i.split(".")[0] for i in os.listdir(root) if i.endswith(".gif") or i.endswith(".mp4")]))
    print(out)
    return out


def make_tileable_horizontal(image, n=3):
    image = image.convert('RGBA')
    img_array = np.array(image)

    height, width, channels = img_array.shape
    third_width = width // n
    if width % n != 0:
        img_array = img_array[:, :third_width*n]
        width = third_width*n

    left = img_array[:, :third_width]
    middle = img_array[:, third_width:-third_width]
    right = img_array[:, -third_width:]

    x = np.linspace(0, np.pi, third_width)
    blend_mask = (np.tile(np.cos(x), (height, 1)) / 2 + 0.5) / 2
    blend_mask = np.expand_dims(blend_mask, axis=2)

    left_blended = left * (1 - blend_mask) + right[:, ::-1] * blend_mask
    right_blended = right * (1 - blend_mask[:,::-1]) + left[:, ::-1] * blend_mask[:,::-1]
    blended_image = np.concatenate((left_blended, middle, right_blended), axis=1)
    return Image.fromarray(blended_image.astype('uint8'))


def make_tileable_vertical(image, n=3):
    image = image.convert('RGBA')
    img_array = np.array(image)

    height, width, channels = img_array.shape
    third_width = height // n
    if height % n != 0:
        img_array = img_array[:third_width*n]
        height = third_width*n

    left = img_array[:third_width]
    middle = img_array[third_width:-third_width]
    right = img_array[-third_width:]

    x = np.linspace(0, np.pi, third_width)
    blend_mask = (np.tile(np.cos(x)[:, None], (1, width)) / 2 + 0.5) / 2
    blend_mask = np.expand_dims(blend_mask, axis=2)

    left_blended = left * (1 - blend_mask) + right[::-1] * blend_mask
    right_blended = right * (1 - blend_mask[::-1]) + left[::-1] * blend_mask[::-1]
    blended_image = np.concatenate((left_blended, middle, right_blended), axis=0)
    return Image.fromarray(blended_image.astype('uint8'))
