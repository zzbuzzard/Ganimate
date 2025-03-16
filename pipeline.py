import gc
import numpy as np
from PIL import Image

from gan import GAN
import upscalers
import remove_bg
import util


def generate_from_config(config, zs, gan: GAN, upscaler: upscalers.Upscaler, remove_bg_session, batch_size, progress=None,
                         writer=None):
    imgs = gan.generate(zs, batch_size, progress, **config, writer=writer)
    count = len(imgs)

    if config["upscale"]:
        upscaler.to(util.device)

        progress((0, count), desc="Upscaling")

        upsampled = []
        for i in range(count):
            x = upscaler.enhance(np.array(imgs[i]))
            upsampled.append(Image.fromarray(x))

            progress((i+1, count), desc="Upscaling")

        imgs = upsampled
        upscaler.cpu()

    if config["bg_remove"]:
        gc.collect()
        progress((0, count), desc="Removing BG")
        bg_removed = []
        for i in range(count):
            bg_removed.append(remove_bg.remove_bg(remove_bg_session, imgs[i]))
            progress((i+1, count), desc="Removing BG")
        imgs = bg_removed

    if config["x_mirror"]:
        imgs = [util.make_tileable_horizontal(im) for im in imgs]
    if config["y_mirror"]:
        imgs = [util.make_tileable_vertical(im) for im in imgs]

    return imgs
