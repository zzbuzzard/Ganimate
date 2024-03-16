import torch
import torchvision.transforms.functional as trf
import numpy as np
from typing import List, Tuple
from PIL import Image
import pytorch_pretrained_biggan as bg
from torch.nn.functional import one_hot

from gan import GAN
from util import *

# kwargs:
#  truncation   (0-1)
#  classes      list of strings, or empty for random class
#  multiplier


class BigGAN(GAN):
    def __init__(self):
        super().__init__("BigGAN",  8)
        self.model = bg.BigGAN.from_pretrained('biggan-deep-256').to(device)

    # Applies BIGGAN but batched
    #  noises : N x latent
    #  classes : N x class
    @torch.no_grad()
    def batched_gen(self, noises, classes, trunc, batch_size=12):
        N = noises.shape[0]
        output = torch.zeros((N, 3, 256, 256))

        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)

            with torch.no_grad():
                n = noises[i:j]
                c = classes[i:j]
                out = self.model(n, c, trunc)

                output[i:j] = out

        return output

    @torch.no_grad()
    def get_z(self, batch_size, classes=tuple(), z_mul=1, class_mul=1, trunc=0.5, **kwargs):
        cs = None
        if len(classes) > 0:
            n = len(classes)
            cs = bg.one_hot_from_names(classes, n)
            if cs is None:
                print("One or more classes not recognised:", classes)
            else:
                cs = torch.from_numpy(cs)
                rnd = torch.softmax(torch.randn(batch_size, n), -1)
                cs = rnd @ cs
        if cs is None:
            cs = one_hot(torch.randint(0, 1000, (batch_size,)), num_classes=1000)

        zs = torch.from_numpy(bg.truncated_noise_sample(truncation=trunc, batch_size=batch_size))

        zs = zs * z_mul
        cs = cs * class_mul

        out = torch.cat((zs, cs), dim=1)
        return out.numpy()

    @torch.no_grad()
    def generate(self, zs, batch_size, trunc=0.5, unclip=True, **kwargs) -> List[Image.Image]:
        assert zs.shape[1] == 1128, f"Invalid shape {zs.shape}"
        zs = torch.from_numpy(zs).to(device)
        z = zs[:, :128]
        c = zs[:, 128:]
        out = self.batched_gen(z, c, trunc, batch_size=batch_size)

        if unclip:
            out = torch.clip(out * .5 + .5, 0, 1)

        return [trf.to_pil_image(i) for i in out.unbind(0)]

