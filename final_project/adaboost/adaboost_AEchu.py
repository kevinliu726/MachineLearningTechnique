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

class AE_chu(nn.Module): # best AutoEncoder
    def __init__(self):
        super(AE_chu, self).__init__()
        
        self.encoder = nn.Sequential( 
            # conv : [(in + 2*pad - dilation(kerenl-1) -1) / stride + 1]
            # 224 X 224
            nn.Conv2d(  3,  32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d( 32,  64, kernel_size=3, stride=2, padding=1),
            # 112 X 112
            nn.Conv2d( 64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            # 56 X 56
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            # 28 X 28
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            # 14 X 14
        )
 
        self.decoder = nn.Sequential( 
            #conv transpose : (in-1)stride-2pad+dilation(kernel-1)+outputpad+1
            nn.ConvTranspose2d( 512,  512, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d( 512,  512, kernel_size=4, stride=2, padding=1),

            nn.ConvTranspose2d( 512,  256, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d( 256,  256, kernel_size=4, stride=2, padding=1),

            nn.ConvTranspose2d( 256,  128, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d( 128,   64, kernel_size=4, stride=2, padding=1),

            nn.ConvTranspose2d(  64,   32, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(  32,    3, kernel_size=4, stride=2, padding=1),

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

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

    model = AE_chu().cuda()
    model.load_state_dict(torch.load(model_fpath))
    model.eval()

    train_code = get_encoded_vector(train_dataset, model)
    valid_code = get_encoded_vector(valid_dataset, model)

    predict_acc(root_dpath, train_code, valid_code)

    return

if __name__ == "__main__":
    batch_size = 32
    main()
