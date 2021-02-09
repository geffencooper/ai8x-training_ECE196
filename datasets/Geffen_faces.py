###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Faces Datasets
"""
from skimage import io, transform
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image
import torch

import ai8x


def geffen_faces_get_datasets(data, load_train=True, load_test=True):
   
    (data_dir, args) = data
    
    training_data_path = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_classifier/train"
    test_data_path = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_classifier/test"

    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=training_data_path,
                                                         transform=train_transform)
        print(train_dataset.classes)
    else:
        train_dataset = None

    # Loading and normalizing test dataset
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=test_data_path,
                                                        transform=test_transform)

    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'GEFFEN_FACES',
        'input': (1, 64, 64),
        'output': ('face', 'no_face'),
        'loader': geffen_faces_get_datasets,
    }
]
