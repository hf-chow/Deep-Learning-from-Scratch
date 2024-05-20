import numpy as np
from PIL import Image
import typing
import h5py

def load_image(img_file: str) -> np.float64:
    img = Image.open(img_file)
    img.load()
    img_data = np.asarray(img, dtype="float64")
    return img_data

def load_hdf5(path):
    f = h5py.File(path, "r")
    data = f['image'][...]
    label = f['label'][...]
    f.close()
    return data, label

PATH="../../data/MNIST/train.hdf5"
x, y = load_hdf5(PATH)

