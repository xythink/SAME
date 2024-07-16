"""Module for dataset related functions"""
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from .flowers import Flowers17
from .imagenet1k import ImageNet1k
from .tiny_images import TinyImages
from .gtsrb import GTSRB
from .emnist import EMNISTLetters, EMNISTDigits
from functools import partial
from .indoor67 import Indoor67

from torch.utils.data import random_split

torch.multiprocessing.set_sharing_strategy("file_system")
num_workers = 4
ds_root = "./data/"


def get_weights(ds, n_classes):
    weights = []
    for _, y in ds:
        if y < n_classes:
            weights.append(1)
        else:
            weights.append(0)
    return weights




def get_dataloaders(
    ds: str = "cifar10",
    batch_size: int = 256,
    augment: bool = False,
    standardize: bool = False,
    shuffle=True,
    n_classes=None,
) -> Tuple[DataLoader, DataLoader]:
    """returns train and test loaders"""

    transform_train_list, transform_test_list = [], []
    if ds in ["tiny_images"]:
        transform_train_list = [transforms.ToPILImage()]
        transform_test_list = [transforms.ToPILImage()]
    elif ds in ["indoor67", "imagenet", "flowers17", "imagenet_tiny","food101", "flowers102"]:
        transform_train_list = [transforms.RandomResizedCrop(224), transforms.Resize(32)]
        transform_test_list = [transforms.RandomResizedCrop(224), transforms.Resize(32)]
    elif ds in ["svhn_28"]:
        transform_train_list = [transforms.Resize(28), transforms.Grayscale()]
        transform_test_list = [transforms.Resize(28), transforms.Grayscale()]
    

    if augment:
        transform_train_list += transforms_augment[ds]

    transform_train_list += [transforms.ToTensor()]
    transform_test_list += [transforms.ToTensor()]

    if standardize:
        transform_train_list += transforms_standardize[ds]
        transform_test_list += transforms_standardize[ds]
    else:
        transform_train_list += transforms_normalize[ds]
        transform_test_list += transforms_normalize[ds]

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    if ds in ["svhn", "svhn_28"]:
        dataset_train = ds_dict[ds](
            root=f"{ds_root}/svhn",
            split="train",
            transform=transform_train,
            download=True,
        )
        dataset_test = ds_dict[ds](
            root=f"{ds_root}/svhn", split="test", transform=transform_test,download=True
        )
    elif ds in ["fake_28"]:
        dataset_train = ds_dict[ds](size=50000, transform=transform_train)
        dataset_test = ds_dict[ds](size=10000, transform=transform_test)
    elif ds in ["food101","flowers102"]:
        dataset_train = ds_dict[ds](
            root=f"{ds_root}/{ds}",
            split="train",
            transform=transform_train,
            download=True,
        )
        dataset_test = ds_dict[ds](
            root=f"{ds_root}/{ds}", split="test", transform=transform_test,download=True
        )
    else:
        if ds in ["imagenet_tiny"]:
            ds = "imagenet"
        dataset_train = ds_dict[ds](
            root=f"{ds_root}/{ds}",
            train=True,
            transform=transform_train,
            download=True,
        )
        dataset_test = ds_dict[ds](
            root=f"{ds_root}/{ds}", train=False, transform=transform_test,
        )

    shuffle_train = shuffle
    shuffle_test = True
    if ds in ["tiny_images", "imagenet", "imagenet_tiny", "indoor67"]:
        shuffle_train = False

    if n_classes is None:
        sampler_train = sampler_test = None
    else:
        shuffle_train = shuffle_test = False
        weights_train = get_weights(dataset_train, n_classes)
        weights_test = get_weights(dataset_test, n_classes)

        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(
            weights_train, len(weights_train)
        )
        sampler_test = torch.utils.data.sampler.WeightedRandomSampler(
            weights_test, len(weights_test)
        )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=False,
        sampler=sampler_train,
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler_test,
    )

    return dataloader_train, dataloader_test
