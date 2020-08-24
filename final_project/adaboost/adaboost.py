import os 
import sys
import pandas as pd
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import class_model

class Flatten(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        return x.view(-1, self.d)

class ToImage(nn.Module):
    def __init__(self, channel, image_size):
        super().__init__()
        self.out_shape = (channel, *image_size)
    
    def forward(self, x):
        return x.view(-1, *self.out_shape)

class AE(nn.Module):
    def __init__(self, img_size=(224, 224)):
        super(AE, self).__init__()

        H = img_size[0]
        W = img_size[1]

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.AvgPool2d(2),

            Flatten(16 * (H // 4) * (W // 4)),
            nn.Linear(16 * (H // 4) * (W // 4), 1024),
        )
 
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 16 * (H // 4) * (W // 4)),
            nn.BatchNorm1d(16 * (H // 4) * (W // 4)),
            nn.ReLU(True),

            ToImage(16, (H // 4, W // 4)),

            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, 5, stride=1, padding=2, bias=False),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x  = self.decoder(x1)
        return x1, x


def get_encoded_vector(dataset, model):
    print("get encoded vector!")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    code = None # extended encoded vector
    for i, data in enumerate(dataloader): #mini-batch
        encoded, decoded = model(torch.FloatTensor(data).cuda()) 
        encoded_vector = encoded.view(data.size()[0], -1).cpu().detach().numpy() #[batch, vector]

        if i == 0:
            code = np.empty((0, np.shape(encoded_vector)[1]))
        code = np.concatenate((code, encoded_vector), axis = 0)

    return code


def predict_acc(root_dpath, train_code, valid_code):
    df = pd.read_csv(os.path.join(root_dpath, "train.csv"), header=0)
    train_label = list(df.label)
    train_label = [ord(y)-ord('A') for y in train_label]
    train_label = np.array(train_label)

    df = pd.read_csv(os.path.join(root_dpath, "validation.csv"), header=0)
    valid_label = list(df.label)
    valid_label = [ord(y)-ord('A') for y in valid_label]
    valid_label = np.array(valid_label)

    print("Adaboost!")
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=25, learning_rate=1.1, random_state=111) 
    clf.fit(train_code, train_label)
    print("train accuracy:", clf.score(train_code, train_label))
    print("valid accuracy:", clf.score(valid_code, valid_label))

    return


# python3 adaboost.py [root] [.pth]
def main():
    root_dpath  = sys.argv[1]
    model_fpath = sys.argv[2]

    train_dataset = class_model.Dataset(os.path.join(root_dpath, "train"), mode="test")
    valid_dataset = class_model.Dataset(os.path.join(root_dpath, "validation"), mode="test")

    model = AE().cuda()
    model.load_state_dict(torch.load(model_fpath))
    model.eval()

    train_code = get_encoded_vector(train_dataset, model)
    valid_code = get_encoded_vector(valid_dataset, model)

    predict_acc(root_dpath, train_code, valid_code)

    return

if __name__ == "__main__":
    batch_size = 32
    main()
