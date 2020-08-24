import os
import sys
import class_model
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import torch.nn as nn

IMG_SIZE = (224,224)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    def __init__(self, img_size=(64, 64)):
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

def _validation(model, valid_dataset):
    model.eval()
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    criterion = nn.MSELoss()
    epoch_loss = 0.0
    with torch.no_grad():
        for data, label in valid_loader:
            img = data.cuda()

            output1, output = model(img)
            loss = criterion(output, img)
            
            epoch_loss += loss.item()
    return epoch_loss / len(valid_loader)

def _training(model, train_dataset):
    model.train()
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    epoch_loss = 0.0
    for data, label in train_loader:
        img = data.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def _train(model, train_dataset, valid_dataset):
    print("Loading training data...")
    
    print("Train!")
    for epoch in range(num_epochs):
        train_loss = _training(model, train_dataset)
        if valid_dataset == None:
            print('epoch [ {} / {}], loss: {:.5f}'.format(epoch+1, num_epochs, train_loss))
        else:
            valid_loss = _validation(model, valid_dataset)
            print('epoch [ {} / {}], train loss: {:.5f}, valid loss: {:.5f}'.format(epoch+1, num_epochs, train_loss, valid_loss))
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), './AE_check/checkpoint_{}.pth'.format(epoch+1))
    return model

def main():
    train_dpath = os.path.join(sys.argv[1], "train") # no / ending
    valid_dpath = os.path.join(sys.argv[1], "validation") # no / endin
    model_fpath = sys.argv[2]

    train_dataset = class_model.Dataset(train_dpath, mode="train")
    valid_dataset = class_model.Dataset(valid_dpath, mode="validation")
    model = AE(img_size = IMG_SIZE).cuda()
    model = _train(model, train_dataset, valid_dataset)
    
    torch.save(model.state_dict(), model_fpath)
    #train_valid_dataset = ConcatDataset([train_dataset, valid_dataset])
    #model = _train(model, train_dataset, valid_dataset = None)

    #torch.save(model.state_dict(), model_fpath)

if __name__ == "__main__":
    seed = 111
    set_random_seed(seed)

    # arguments
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 500
    weight_decay = 5e-6

    main()
