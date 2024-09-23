import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])


def load_training(img_size, root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([img_size, img_size]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir, 'trainset'), transform=transform)
    if dir == 'template26':
        num_workers = 1
    else:
        num_workers = 0
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=num_workers)
    return train_loader


def load_testing(img_size, root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([img_size, img_size]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir, 'testset'), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)

    return test_loader


def load_training_mixture(img_size, root_path, source_dir, target_dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([img_size, img_size]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    source_data = datasets.ImageFolder(root=os.path.join(root_path, source_dir, 'trainset'), transform=transform)
    target_data = datasets.ImageFolder(root=os.path.join(root_path, target_dir, 'trainset'), transform=transform)
    data = ConcatDataset([source_data, target_data])
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    return train_loader
