from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np


# TODO improve performance with vectorization or Cython
def normalize_2d(dataset, batch_size=1024, shuffle=False):
    # Normalize the 0-255 RGB values to 0-1
    # A 2d RGB image actually has a 3D tensor = [(RGB), W, H]
    # We would like to extract the mean R, G and B  values across the W and H
    # And recall that the first index of dataloader is the number of image in the batch
    # Using numpy's syntax, we will compile mean and std across axis = (0, 2, 3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    batch_means = []
    batch_stds = []
    for batch, (image, target) in enumerate(dataloader):
        sample_means = []
        sample_stds = []
        for sample in range(image.shape[0]):
            current_sample = image[sample, :, :, :].numpy()
            channel_means =  []
            channel_stds = []
            for channel in range(current_sample.shape[0]):
                channel_means.append(np.mean(current_sample[channel,:, :]))
                channel_stds.append(np.std(current_sample[channel, :, :]))
            sample_means.append(channel_means)
            sample_stds.append(channel_stds)

        batch_means.append(np.mean(np.array(sample_means), axis=0))
        batch_stds.append(np.mean(np.array(sample_stds), axis=0))
    means = list(np.mean(batch_means, axis = 0))
    stds = list(np.mean(batch_stds, axis = 0))

    return means, stds

def main():
    PATH = "../../data/CIFAR100/"
    cifar = datasets.CIFAR100(PATH, download=False, transform=ToTensor())
    means, stds = normalize_2d(cifar)
    print(means, stds)

if __name__ == "__main__":
    main()
