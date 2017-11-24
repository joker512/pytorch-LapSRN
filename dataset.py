import torch.utils.data as data
import torch
import numpy as np
import h5py
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random


def convert_to_numpy(image, torch_rep=True):
    ar = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3)
    return ar.transpose(2, 0, 1) if torch_rep else ar


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.label_x2 = hf.get("label_x2")
        self.label_x4 = hf.get("label_x4")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.label_x2[index,:,:,:]).float(), torch.from_numpy(self.label_x4[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]


class DatasetFromFolder(data.Dataset):
    def __init__(self, path, batch_size):
        super(DatasetFromFolder, self).__init__()
        self.path = path
        self.batch_size = batch_size
        self.images = os.listdir(path)
        self.label_size = 128, 128
        self.stride = 64
        self.scale = 4
        self.downsizes = [1, 0.7, 0.5]

    def crop_grid(image):
        downsize = random.choice(self.downsizes)
        if downsize != 1:
            image = image.resize(self.label_size, Image.BICUBIC)
        width, height = image.size
        assert width >= self.label_size[0] and height >= self.label_size[1]

        patch_x, patch_y = random.randint(0, width // self.stride - 1), random.randint(0, height // self.stride - 1)
        margin_left = width % self.stride // 2 + patch_x * self.stride
        margin_down = patch_y * self.stride

        return image.crop((margin_left, margin_down, margin_left + self.label_size[0], margin_down + self.label_size[1]))

    def crop_random(self, image):
        assert self.label_size[0] == self.label_size[1]
        width, height = image.size
        assert self.label_size[0] * 2 <= width and self.label_size[1] * 2 <= height

        patch_size = (random.randint(self.label_size[0], self.label_size[0] * 2), ) * 2
        patch_x, patch_y = tuple(random.randint(0, im - pt) for im, pt in zip(image.size, patch_size))

        crop_image = image.crop((patch_x, patch_y, patch_x + patch_size[0], patch_y + patch_size[1]))
        return crop_image.resize(self.label_size, Image.BICUBIC)

    def __getitem__(self, index):
        if index % self.batch_size == 0:
            image_name = self.images[index]
            self.image = Image.open(os.path.join(self.path, image_name))
            self.image = self.image.convert('RGB')

        crop_image = self.crop_random(self.image)
        label_x4 = convert_to_numpy(crop_image)
        label_x2 = convert_to_numpy(crop_image.resize(tuple(ti // self.scale * 2 for ti in self.label_size), Image.BICUBIC))
        data = convert_to_numpy(crop_image.resize(tuple(ti // self.scale for ti in self.label_size), Image.BICUBIC))

        return torch.from_numpy(data).float(), torch.from_numpy(label_x2).float(), torch.from_numpy(label_x4).float()

    def __len__(self):
        return len(self.images) * self.batch_size
