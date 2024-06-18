import gc
import numpy as np
from PIL import Image

from gan import GAN
import upscalers
import remove_bg
import util


def generate_from_config(config, zs, gan: GAN, upscaler: upscalers.Upscaler, remove_bg_session, batch_size, ctr_callback=None):
    if ctr_callback is None:
        ctr_callback = lambda *args, **kwargs: None

    ctr_callback("GAN")
    imgs = gan.generate(zs, batch_size, **config)
    count = len(imgs)

    itered = False

    if config["upscale"]:
        upscaler.to(util.device)
        ctr_callback("Upscaling")

        upsampled = []
        for i in range(count):
            x = upscaler.enhance(np.array(imgs[i]))
            upsampled.append(Image.fromarray(x))
            ctr_callback()

        imgs = upsampled
        upscaler.cpu()

        itered = True

    if config["bg_remove"]:
        gc.collect()
        ctr_callback("Removing BG")
        bg_removed = []
        for i in range(count):
            bg_removed.append(remove_bg.remove_bg(remove_bg_session, imgs[i]))
            if not itered:
                ctr_callback()
        imgs = bg_removed
        itered = True

    if config["x_mirror"]:
        imgs = [util.make_tileable_horizontal(im) for im in imgs]
    if config["y_mirror"]:
        imgs = [util.make_tileable_vertical(im) for im in imgs]

    if not itered:
        ctr_callback(count)

    return imgs
