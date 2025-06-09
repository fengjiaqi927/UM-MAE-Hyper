# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageListFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000

        assert ann_file is not None
        print('load info from', ann_file)

        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            cut = elem.split(' ')
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()

        print('load finish')


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # TODO modify your own dataset here
    folder = os.path.join(args.data_path, 'train' if is_train else 'val')
    ann_file = os.path.join(args.data_path, 'train.txt' if is_train else 'val.txt')
    dataset = ImageListFolder(folder, transform=transform, ann_file=ann_file)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)



from torch.utils.data.dataset import Dataset

from typing import Any, Optional, List
import random
import numpy as np
import csv
import torch


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)





class HySpecNet11k(SatelliteDataset):
    """
    Dataset:
        HySpecNet-11k
    Authors:
        Martin Hermann Paul Fuchs
        Begüm Demir
    Related Paper:
        HySpecNet-11k: A Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods
        https://arxiv.org/abs/2306.00385
    Cite: TODO
        @misc{fuchs2023hyspecnet11k,
            title={HySpecNet-11k: A Large-Scale Hyperspectral Dataset for Benchmarking Learning-Based Hyperspectral Image Compression Methods}, 
            author={Martin Hermann Paul Fuchs and Begüm Demir},
            year={2023},
            eprint={2306.00385},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
        }

    Folder Structure:
        - root_dir/
            - patches/
                - tile_001/
                    - tile_001-patch_01/
                        - tile_001-patch_01-DATA.npy
                        - tile_001-patch_01-QL_PIXELMASK.TIF
                        - tile_001-patch_01-QL_QUALITY_CIRRUS.TIF
                        - tile_001-patch_01-QL_QUALITY_CLASSES.TIF
                        - tile_001-patch_01-QL_QUALITY_CLOUD.TIF
                        - tile_001-patch_01-QL_QUALITY_CLOUDSHADOW.TIF
                        - tile_001-patch_01-QL_QUALITY_HAZE.TIF
                        - tile_001-patch_01-QL_QUALITY_SNOW.TIF
                        - tile_001-patch_01-QL_QUALITY_TESTFLAGS.TIF
                        - tile_001-patch_01-QL_SWIR.TIF
                        - tile_001-patch_01-QL_VNIR.TIF
                        - tile_001-patch_01-SPECTRAL_IMAGE.TIF
                        - tile_001-patch_01-THUMBNAIL.jpg
                    - tile_001-patch_02/
                        - ...
                    - ...
                - tile_002/
                    - ...
                - ...
            - splits/
                - easy/
                    - test.csv
                    - train.csv
                    - val.csv
                - hard/
                    - test.csv
                    - train.csv
                    - val.csv
                - ...
            - ...
    """
    def __init__(self, 
                root_dir, 
                mode: str = "easy", 
                split: str = "train", 
                transform=None,
                saved_bands_num: Optional[List[int]] = None):

        super().__init__(in_c=202)
        self.root_dir = root_dir

        self.saved_bands_num = saved_bands_num
        if self.saved_bands_num is not None:
            self.in_c = int(sum(self.saved_bands_num))

        self.csv_path = os.path.join(self.root_dir, "splits", mode, f"{split}.csv")
        with open(self.csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.npy_paths = sum(csv_data, [])
        self.npy_paths = [os.path.join(self.root_dir, "patches", x) for x in self.npy_paths]

        self.transform = transform

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, index):
        # get full numpy path
        npy_path = self.npy_paths[index]
        # read numpy data
        img = np.load(npy_path)
        # convert numpy array to pytorch tensor
        img = torch.from_numpy(img)
        # apply transformations

        if self.saved_bands_num is not None:
            watershed = int(img.shape[0]/random.uniform(2.8, 3.2))
            saved_bands = sorted(random.sample(range(watershed), self.saved_bands_num[0])+random.sample(range(watershed,img.shape[0]), self.saved_bands_num[1]))
            keep_idxs = [i for i in range(img.shape[0]) if i in saved_bands]
            img = img[keep_idxs, :, :]

        if self.transform:
            img, mask_UM = self.transform(img)
            return img, mask_UM
        return img, 'unlabeled'
            

