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
import os

import ai8x


class FacePointsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        print(self.landmarks_frame)
        print(len(self.landmarks_frame))
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        print(self.landmarks_frame)
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # the img name is the last column
        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[idx, 8])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx,:-1] # don't include image name
        landmarks = np.array(landmarks).astype('float')
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 1) # change to coordinate columns
        #landmarks = np.squeeze(landmarks, axis = 1)
        
        image = torch.Tensor(image).float()
        image = image.unsqueeze(0)
        landmarks = torch.from_numpy(landmarks).float()
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, landmarks
    

def geffen_faces_get_datasets(data, load_train=True, load_test=True):
   
    (data_dir, args) = data
    
    train_data_dir = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_points/train"
    test_data_dir = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_points/test"
    test_csv_file = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_points/test/test_points.csv"
    train_csv_file = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_points/train/train_points.csv"
    
    transform = transforms.Compose([
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = FacePointsDataset(csv_file=train_csv_file, root_dir=train_data_dir, transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = FacePointsDataset(csv_file=test_csv_file, root_dir=test_data_dir, transform=transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


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
        bb = np.array(bb).astype('float')/2
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 1) # change to coordinate columns
        #landmarks = np.squeeze(landmarks, axis = 1)
        
        image = torch.Tensor(image).float()
        image = image.unsqueeze(0)
        bb = torch.from_numpy(bb).float()
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, bb
  

def geffen_bb_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data
    
    train_data_dir = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_box/train"
    test_data_dir = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_box/test"
    test_csv_file = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_box/test/test_bb.csv"
    train_csv_file = "/home/geffen/Documents/Source/AI/ai8x-training/data/face_box/train/train_bb.csv"
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
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
        'name': 'geffen_points',
        'input': (1, 64, 64),
        'output': ('coords'),
        'regression': True,
        'loader': geffen_faces_get_datasets,
    },
    {
        'name': 'geffen_bb',
        'input': (1, 64, 64),
        'output': ('coords'),
        'regression': True,
        'loader': geffen_bb_get_datasets,
    }
]
