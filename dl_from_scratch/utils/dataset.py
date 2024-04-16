from os.path import abspath, join

from torchvision import datasets
from tqdm import tqdm

class MNIST:
    def __init__(self, download: bool = False) -> None:
        if download:
            self.download()

    def download(self) -> None:
        # Since the MNIST dataset is not strictly open source, this function is a wrapper of torch.dataset.MNIST
        DATA_PATH = abspath(join(__file__, "../../data/MNIST/"))
        data = datasets.MNIST(root=DATA_PATH, download=True)
        for i, (img, _) in tqdm(enumerate(data)):
            img.save(f"{DATA_PATH}/{i}.png")

mnist = MNIST(download=True)
