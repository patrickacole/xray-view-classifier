import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip
from PIL import Image

# custom
from .labels import *

class CheXpertDataset(Dataset):
    """
    Load images and corresponding labels for CheXpert dataset
    This particular dataset only predicts Frontal / Lateral view
    """
    def __init__(self, data_path, img_size=256, train=True, transforms=None):
        """
        Initialize data set by loading in all of the file names and labels
        @param data_path : path to CheXpert data
        @param img_size : desired height and width of images
        @param train : bool to tell whether to load train data or test
        @param transforms : list of transforms to be applied to image
        """
        if not os.path.exists(data_path):
            raise IOError(f'Path given for CheXpertDataset {data_path} does not exist...')

        self.data_path = data_path
        csvfile = "train.csv" if train else "valid.csv"
        self.df = pd.read_csv(os.path.join(data_path, csvfile))

        # compute better data split for training data
        if train:
            # split dataset into frontal and lateral
            frontal = self.df[self.df['Frontal/Lateral'] == 'Frontal']
            lateral = self.df[self.df['Frontal/Lateral'] == 'Lateral']

            # find out min number of samples
            min_length = min(len(frontal), len(lateral))

            # sample from based on the min number of samples
            if len(frontal) > min_length:
                num_samples = min(len(frontal), int(1.5 * min_length))
                frontal = self.df[self.df['Frontal/Lateral'] == 'Frontal'].sample(num_samples)
            else:
                num_samples = min(len(lateral), int(1.5 * min_length))
                lateral = self.df[self.df['Frontal/Lateral'] == 'Lateral'].sample(num_samples)

            # reconstruct new data frame
            self.df = frontal.append(lateral)

        print("{} stats..".format("train" if train else "valid"))
        print(self.df['Frontal/Lateral'].value_counts(), '\n')

        self.img_size = img_size
        self.transforms = transforms
        self.flip = RandomHorizontalFlip(p=0.2)

    def __len__(self):
        """
        Get length of dataset
        @return len : length of dataset
        """
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Gets data at a certain index
        @param idx : idx of data desired
        """
        findings = self.df.iloc[idx]
        label = torch.zeros(1, dtype=torch.float32)
        label[0] = Labels.convert(findings['Frontal/Lateral'])

        imgfile = "/".join(findings['Path'].split('/')[1:])
        xray = Image.open(os.path.join(self.data_path, imgfile))
        xray = xray.resize((self.img_size, self.img_size), Image.LANCZOS)
        xray = xray.convert('L')

        # if lateral radiograph flip randomly
        if label[0] == 0:
            xray = self.flip(xray)
        if self.transforms:
            xray = self.transforms(xray)
        return xray, label

    def at(self, idx):
        """
        Gets directory name for a certain index
        @param idx : idx of data directory desired
        """
        findings = self.df.iloc[idx]
        imgfile = "/".join(findings['Path'].split('/')[1:])
        return imgfile

if __name__ == "__main__":
    from torchvision import transforms
    import matplotlib.pyplot as plt

    print('Testing CheXpert dataset...')
    path = os.path.expanduser("~/Downloads/CheXpert-v1.0-small/")
    transform = [transforms.ToTensor(), transforms.Normalize((0.502845,), (0.290049,))]
    dataset = CheXpertDataset(path, transforms=transforms.Compose(transform))
    randidx = np.random.randint(len(dataset))
    print(len(dataset))
    print(dataset[randidx][0].shape, dataset[randidx][1].shape)
    print(dataset[randidx][1])
    print(dataset.at(randidx))

    plt.imshow(dataset[randidx][0][0], cmap='gray')
    plt.show()
