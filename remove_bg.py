from PIL import Image
from rembg import remove, new_session


def get_remove_bg_model():
    return new_session(model_name="u2net", providers=["CPUExecutionProvider"])


def remove_bg(model, im: Image.Image) -> Image.Image:
    return remove(im, session=model)
