""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        #if transform is given, we transoform data using
        with open(os.path.join(path, 'train'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['fine_labels'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, 'test'), 'rb') as cifar100:
            self.data = pickle.load(cifar100, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.data['data'.encode()])

    def __getitem__(self, index):
        label = self.data['fine_labels'.encode()][index]
        r = self.data['data'.encode()][index, :1024].reshape(32, 32)
        g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
        b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image

class PairedDataset(Dataset):
    def __init__(self, pairs, labels, transform_vis=None, transform_trans=None):
        self.pairs = pairs
        self.labels = labels
        self.transform_vis = transform_vis
        self.transform_trans = transform_trans

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vis_path, trans_path = self.pairs[idx]
        label = self.labels[idx]
        vis_image = Image.open(vis_path).convert('RGB')
        trans_image = Image.open(trans_path).convert('RGB')

        if self.transform_vis and self.transform_trans:
            vis_image = self.transform_vis(vis_image)
            trans_image = self.transform_trans(trans_image)

        return vis_image, trans_image, label
    