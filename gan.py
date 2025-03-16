from abc import abstractmethod, ABC
import numpy as np
from typing import List, Tuple
from PIL import Image

import util


class GAN(ABC):
    def __init__(self, name: str, default_batch_size: int, res: Tuple[int, int]):
        self.name = name
        if util.get_tile_mode() and not self.name.endswith("_t"):
            self.name = self.name + "_t"
        self.default_batch_size = default_batch_size

        self.res = res

        self.mean_latent = 0
        self.std_latent = 1

    @abstractmethod
    def get_z(self, batch_size, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def generate(self, z, batch_size, progress=None, **kwargs) -> List[Image.Image]:
        raise NotImplementedError

    # @abstractmethod
    # def gen_args(self) -> List[Tuple]:
    #     """Returns e.g. [(mul, 'float', 0, 1), ()]"""
    #     raise NotImplementedError
