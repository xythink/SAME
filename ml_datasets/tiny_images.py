import numpy as np
import torch
from bisect import bisect_left
import os.path as osp


class TinyImages(torch.utils.data.Dataset):
    def __init__(
        self,
        train=True,
        transform=None,
        exclude_cifar=True,
        root="./data",
        **kwargs
    ):

        data_file = open(osp.join(root, "tiny_images.bin"), "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype="uint8").reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0  # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open(osp.join(root, "80mn_cifar_idxs.txt"), "r") as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(0)

    def __len__(self):
        return 79302017 - len(self.cifar_idxs)

