import numpy as np
from PIL import Image
import typing

def load_image(img_file: str) -> np.float64:
    img = Image.open(img_file)
    img.load()
    img_data = np.asarray(img, dtype="float64")
    return img_data
