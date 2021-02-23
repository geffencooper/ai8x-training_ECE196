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
import pandas as pd

import ai8x

# face dataset of 70k images of faces and 70k images of random objects and indoor scenes without faces
# 1x90x90 images can pass through the CNN with 'ping ponging'
def face_classifier_get_datasets_80x80(data, load_train=True, load_test=True):
   
    (data_dir, args) = data
    
    training_data_path = "/home/geffen/Desktop/Face_Detector/assemble_face_dataset_utils/face_classifier_dataset/train/"
    test_data_path = "/home/geffen/Desktop/Face_Detector/assemble_face_dataset_utils/face_classifier_dataset/test/"

    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(degrees=5,scale=(0.75,1.25),translate=(0.25,0.25)),
            #torchvision.transforms.RandomCrop(90),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=training_data_path,
                                                         transform=train_transform)
        print(train_dataset.classes)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=test_data_path,
                                                        transform=test_transform)

    else:
        test_dataset = None

    return train_dataset, test_dataset


# face dataset of 70k images of faces and 70k images of random objects and indoor scenes without faces
# 1x128x128 may require streaming
def face_classifier_get_datasets_128x128(data, load_train=True, load_test=True):
   
    (data_dir, args) = data
    
    training_data_path = "/home/geffen/Desktop/Face_Detector/assemble_face_dataset_utils/face_classifier_dataset/train/"
    test_data_path = "/home/geffen/Desktop/Face_Detector/assemble_face_dataset_utils/face_classifier_dataset/test/"

    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandomAffine(degrees=5,scale=(0.85,1.15),translate=(0.15,0.15))
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=training_data_path,
                                                         transform=train_transform)
        print(train_dataset.classes)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=test_data_path,
                                                        transform=test_transform)

    else:
        test_dataset = None

    return train_dataset, test_dataset




# dataset of faces including a bounding box (x,y,w,h)
# this requires custom __len__ and __getitem__ functions
class BBDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.bb_frame = pd.read_csv(csv_file)
        print(self.bb_frame)
        print(len(self.bb_frame))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        print(self.bb_frame)
        return len(self.bb_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # the img name is the last column
        img_name = os.path.join(self.root_dir,self.bb_frame.iloc[idx, 0])
        image = io.imread(img_name)
        bb = self.bb_frame.iloc[idx,1:] # don't include image name
        bb = np.array(bb).astype('float')
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 1) # change to coordinate columns
        #landmarks = np.squeeze(landmarks, axis = 1)
        
        image = torch.Tensor(image).float()
        image = image.unsqueeze(0)
        bb = torch.from_numpy(bb).float()
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, bb
  

def bb_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    
    train_data_dir = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_box/train"
    test_data_dir = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_box/test"
    test_csv_file = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_box/test/test_bb.csv"
    train_csv_file = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_box/train/train_bb.csv"
    
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = BBDataset(csv_file=train_csv_file, root_dir=train_data_dir, transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = BBDataset(csv_file=test_csv_file, root_dir=test_data_dir, transform=transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset




datasets = [
    {
        'name': 'faces_and_non_faces_80',
        'input': (1, 80, 80),
        'output': ('face', 'no_face'),
        'loader': face_classifier_get_datasets_80x80,
    },
    {
        'name': 'faces_and_non_faces_128',
        'input': (1, 128, 128),
        'output': ('face', 'no_face'),
        'loader': face_classifier_get_datasets_128x128,
    },
    {
        'name': 'bb',
        'input': (1, 80, 80),
        'output': ('coords'),
        'regression': True,
        'loader': bb_get_datasets,
    }
]
