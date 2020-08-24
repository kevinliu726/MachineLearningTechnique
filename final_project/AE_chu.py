import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
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

def train(dataset, model):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epoch):
        epoch_loss = 0
        for data, _ in dataloader: #mini-batch
            data = data.cuda()

            encoded, decoded = model(data)
            loss = criterion(decoded, data)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, epoch_loss/len(dataloader)))

        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), './AE_chu_check/checkpoint_{}.pth'.format(epoch+1))

    return model


# python3 AE_chu.py [root] [.pth]
def main():
    root_dpath  = sys.argv[1]
    model_fpath = sys.argv[2]

    train_dataset = class_model.Dataset(os.path.join(root_dpath, "train"), "train")
    valid_dataset = class_model.Dataset(os.path.join(root_dpath, "validation"), "train")
    train_valid_dataset = ConcatDataset([train_dataset, valid_dataset])


    model = AE_chu().cuda()
    model = train(train_valid_dataset, model)
    torch.save(model.state_dict(), model_fpath)

    return


if __name__ == "__main__":

    random.seed(111)
    np.random.seed(111)
    torch.manual_seed(111)
    torch.cuda.manual_seed(111)
    torch.cuda.manual_seed_all(111)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #argument
    n_epoch = 250
    lr = 1e-5
    weight_decay = 5e-6
    batch_size = 32

    
    main()
