import dl_from_scratch
import pytest

def test_read_img():
    SAMPLE_PATH = "tests/samples/MNIST0.png"
    assert type(dl_from_scratch.utils.reader.read_img(SAMPLE_PATH)) == np.float64
