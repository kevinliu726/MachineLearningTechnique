import pandas as pd
import os
import cv2
from enum import Enum
import numpy as np
import torch
import torchvision.transforms
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, mode):
        image_dir = sorted(os.listdir(folder_path))
        x = np.zeros((len(image_dir), 224, 224, 3), dtype=np.uint8)
        y = np.zeros((len(image_dir)), dtype=np.uint8)

        if mode == "train" or mode == "validation":
            df = pd.read_csv(folder_path+".csv", header=0)
            label_dict = dict(zip(list(df.image_id), list(df.label)))

        filename_dict = {}
        for i, filename in enumerate(image_dir):
            img = cv2.imread(os.path.join(folder_path, filename))
            x[i, :, :] = cv2.resize(img,(224, 224))
            filename_dict[i] = filename
            if mode == "train" or mode == "validation":
                y[i] = ord(label_dict[filename])-ord('A')

        print("read image file to np.array done")

        if mode == "train":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(45), #隨機旋轉圖片
                torchvision.transforms.RandomAffine( degrees = 0, translate = (0.05, 0.05), scale = (0.9,1.1), shear = 5, resample = False, fillcolor = 0),

                torchvision.transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)    
                torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) 
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(), #[0,1]
                torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) 
            ])

        # instance variable
        self.x = x
        if mode == "train" or mode == "validation":
            self.y = torch.LongTensor(y)
        self.transform = transform
        self.mode = mode
        self.filename_dict = filename_dict

    def __len__(self): #total size of dataset
        return len(self.x)

    def __getitem__(self, index):
        if self.transform is not None:
            x = self.transform(self.x[index])

        if self.mode == "train" or self.mode == "validation":
            y = self.y[index]
            return x, y
        else:
            return x


class CNN(nn.Module):

    def __init__(self):
        super( CNN, self).__init__()

        # 3 channel 224 X 224
        self.layer0 = nn.Sequential(
                nn.Conv2d( 3, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0) #/2
                )
        # 8 channel 112 X 112
        self.layer1 = nn.Sequential(
                nn.Conv2d( 8, 16, kernel_size=3, stride=1, padding=0), #-2
                nn.BatchNorm2d(16),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=1, padding=1) #+2-1
                )
        # 16 channel 111 X 111
        self.layer2 = nn.Sequential(
                nn.Conv2d( 16, 32, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=1) #/2
                )
        # 32 channel 56 X 56
        self.layer3 = nn.Sequential(
                nn.Conv2d( 32, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0) #/2
                )
        # 256 channel 28 X 28
        self.layer4 = nn.Sequential(
                nn.Conv2d( 256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0)
                )
        # 512 channel 14 X 14
        self.layer5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0),
                )
        # 512 channel 7 X 7
        self.layer6 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0),
                )
        #512 channel 3 X 3

        self.fc = nn.Sequential(
                nn.Linear(512*3*3, 1024),
                nn.PReLU(),
                nn.Linear(1024, 512),
                nn.PReLU(),
                nn.Linear(512, 3)
                )

        self.drop_out = nn.Dropout()


    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size()[0], -1)
        
        out = self.drop_out(out)
        out = self.fc(out)
        return out

class kCNN(nn.Module):
    def __init__(self):
        super( kCNN, self).__init__()

        # 3 channel 224 X 224
        self.layer0 = nn.Sequential(
                nn.Conv2d( 3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0) #/2
                )
        # 32 channel 112 X 112
        self.layer1 = nn.Sequential(
                nn.Conv2d( 32, 64, kernel_size=3, stride=1, padding=0), #-2
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=1, padding=1) #+2-1
                )
        # 64 channel 111 X 111
        self.layer2 = nn.Sequential(
                nn.Conv2d( 64, 128, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(128),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=1) #/2
                )
        # 128 channel 56 X 56
        self.layer3 = nn.Sequential(
                nn.Conv2d( 128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0) #/2
                )
        # 256 channel 28 X 28
        self.layer4 = nn.Sequential(
                nn.Conv2d( 256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0)
                )
        # 512 channel 14 X 14
        self.layer5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0),
                )
        # 512 channel 7 X 7
        self.layer6 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=1, padding=1),
                )
         # 512 channel 6 X 6
        self.layer7 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0),
                )

        #512 channel 3 X 3

        self.fc = nn.Sequential(
                nn.Linear(512*3*3, 1024),
                nn.PReLU(),
                nn.Linear(1024, 512),
                nn.PReLU(),
                nn.Linear(512, 256),
                nn.PReLU(),
                #nn.Linear(256, 128),#新加
                #nn.PReLU(),#新加
                #nn.Linear(128, 64),#新加
                #nn.PReLU(),#新加
                nn.Linear(256, 3),
                )

        self.drop_out = nn.Dropout()


    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.view(out.size()[0], -1)
        
        out = self.drop_out(out)
        out = self.fc(out)
        return out

class fcCNN(nn.Module):
    def __init__(self):
        super( fcCNN, self).__init__()

        # 3 channel 224 X 224
        self.layer0 = nn.Sequential(
                nn.Conv2d( 3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0) #/2
                )
        # 32 channel 112 X 112
        self.layer1 = nn.Sequential(
                nn.Conv2d( 32, 64, kernel_size=3, stride=1, padding=0), #-2
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=1, padding=1) #+2-1
                )
        # 64 channel 111 X 111
        self.layer2 = nn.Sequential(
                nn.Conv2d( 64, 128, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(128),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=1) #/2
                )
        # 128 channel 56 X 56
        self.layer3 = nn.Sequential(
                nn.Conv2d( 128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0) #/2
                )
        # 256 channel 28 X 28
        self.layer4 = nn.Sequential(
                nn.Conv2d( 256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0)
                )
        # 512 channel 14 X 14
        self.layer5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0),
                )
        # 512 channel 7 X 7
        self.layer6 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=1, padding=1),
                )
         # 512 channel 6 X 6
        self.layer7 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d( kernel_size=2, stride=2, padding=0),
                )

        #512 channel 3 X 3

        self.fc = nn.Sequential(
                nn.Linear(512*3*3, 1024),
                nn.PReLU(),
                nn.Linear(1024, 512),
                nn.PReLU(),
                nn.Linear(512, 256),
                nn.PReLU(),
                nn.Linear(256, 128),#新加
                nn.PReLU(),#新加
                nn.Linear(128, 64),#新加
                nn.PReLU(),#新加
                nn.Linear(64, 3),
                )

        self.drop_out = nn.Dropout()


    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.view(out.size()[0], -1)
        
        out = self.drop_out(out)
        out = self.fc(out)
        return out
