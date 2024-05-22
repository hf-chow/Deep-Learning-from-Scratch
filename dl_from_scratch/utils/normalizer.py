from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

def normalize_2d(dataset):
    # Normalize the 0-255 RGB values to 0-1
    # A 2d RGB image actually has a 3D tensor = [(RGB), W, H]
    # We would like to extract the mean R, G and B  values across the W and H
    # And recall that the first index of dataloader is the number of image in the batch
    # Using numpy's syntax, we will compile mean and std across axis = (0, 2, 3)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)

def main():
    PATH = "../../data/CIFAR100/"
    cifar = datasets.CIFAR100(PATH, download=False, transform=ToTensor())
    dataloader = DataLoader(cifar, batch_size=1024, shuffle=False)

    means = []
    stds = []
    image_means = []
    image_stds = []

    for i, (image, target) in enumerate(dataloader):
        np_image = image.numpy()
        np_image_R = np_image[1][0]
        np_image_G = np_image[1][1]
        np_image_B = np_image[1][2]
        channels = (np_image_R, np_image_G, np_image_B)
        for channel in channels:
            channel_means = np.mean(channel)
            channel_stds = np.std(channel)
            image_means.append(channel_means)
            image_stds.append(channel_stds)
            print(len(image_means))
        means.append(image_means)
        stds.append(image_stds)

    print(len(means))
    print(len(means[0]))

           



if __name__ == "__main__":
    main()
