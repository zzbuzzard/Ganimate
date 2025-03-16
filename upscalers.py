from abc import abstractmethod, ABC
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np


class Upscaler(ABC):
    @abstractmethod
    def enhance(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def cpu(self):
        raise NotImplementedError

    @abstractmethod
    def to(self, device):
        raise NotImplementedError


class RealESRGAN(Upscaler):
    def __init__(self, upsampler):
        self.upsampler = upsampler

    def enhance(self, img: np.ndarray) -> np.ndarray:
        return self.upsampler.enhance(img, outscale=4)[0]

    def cpu(self):
        self.upsampler.model.cpu()
        return self

    def to(self, device):
        self.upsampler.model.to(device)
        return self


def get_real_esrgan(name="RealESRGAN_x4plus.pth"):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    path = f"models/{name}"

    upsampler = RealESRGANer(
        scale=4,
        model_path=path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        gpu_id='0'
    )
    return RealESRGAN(upsampler)
