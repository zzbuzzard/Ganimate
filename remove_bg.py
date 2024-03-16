import numpy as np
from skimage import io
import torch
import os
from PIL import Image
# from briarmbg import BriaRMBG
# from utilities import preprocess_image, postprocess_image
from rembg import remove, new_session
import onnxruntime as ort

import util


def get_remove_bg_model():
    return new_session(model_name="u2net", providers=["CPUExecutionProvider"])


def remove_bg(model, im: Image.Image) -> Image.Image:
    return remove(im, session=model)


# def get_remove_bg_model():
#     return BriaRMBG.from_pretrained("briaai/RMBG-1.4").to(util.device).eval()
#
#
# def remove_bg(model, orig_im: Image.Image) -> Image.Image:
#     model_input_size = (512, 512)
#     im = np.array(orig_im)
#     orig_im_size = im.shape[:2]
#     image = preprocess_image(im, model_input_size).to(util.device)
#     result = model(image)
#     result = postprocess_image(result[0][0], orig_im_size)
#
#     alpha_im = Image.fromarray(result)
#     no_bg_image = Image.new("RGBA", alpha_im.size, (0,0,0,0))
#     no_bg_image.paste(orig_im, mask=alpha_im)
#     return no_bg_image
