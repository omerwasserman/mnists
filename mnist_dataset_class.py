import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image

class MnistDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        #self.images = torch.Tensor(images)
        self.images = images
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            #x = Image.fromarray(self.images[idx].astype(np.uint8))
            x = x.transpose(1, 2, 0)
            x = self.transform(x)

        return x, y

    def shape(self):
        return self.images.shape
