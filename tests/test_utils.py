from dl_from_scratch.utils import dataloader

import numpy as np
import pytest

print(np.shape(dataloader.load_image("tests/samples/0.png")))

def test_read_img():
    SAMPLE_IMAGE = "tests/samples/0.png"
    assert type(dataloader.load_image(SAMPLE_SAMPLE)) == np.ndarray
    assert dataloader.load_image(SAMPLE_SAMPLE) == (20, 20)
