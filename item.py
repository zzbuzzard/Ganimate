import numpy as np
import pickle
import util
import shutil
from PIL import Image
import os
from os.path import join


class UnsavedItem:
    """
    Stores data in

    gens/
      [idx].pickle
      [idx].png
      [idx].npy
    """
    def __init__(self, idx, z, img, config):
        self.idx = idx
        root = f"{util.path_unsaved}/"

        # Save config
        pickle.dump(config, open(root+f"{idx}.pkl", "wb"))

        # Save zs
        np.save(root+f"{idx}.npy", z)

        # Save image
        img.save(root+f"{idx}.png")

        self.img_path = root+f"{idx}.png"

    def to_item(self):
        root = f"{util.path_unsaved}/"
        new_idx = util.get_next_item_idx()
        new_root = f"{util.path_saved}/{new_idx}/"
        os.mkdir(new_root)

        shutil.copy(root+f"{self.idx}.pkl", new_root+f"config.pkl")
        shutil.copy(root+f"{self.idx}.png", new_root+f"im.png")
        shutil.copy(root+f"{self.idx}.npy", new_root+f"z.npy")

        return Item(new_idx)


class Item:
    """
    Stores data in

    items/
     name/
        config.pickle
        z.npy
        anim1.npy = list of zs
        anim1.gif = anim
        anim1/    = list of images
    """

    def __init__(self, idx: str):
        self.idx = idx
        self.root = join(util.path_saved, str(idx))
        self.img_path = join(self.root, "im.png")

        self.z = np.load(join(self.root, "z.npy"))
        self.config = pickle.load(open(join(self.root, "config.pkl"), "rb"))
