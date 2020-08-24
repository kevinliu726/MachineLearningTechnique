import numpy as np
import random
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import matplotlib.pyplot as plt
import matplotlib.path as mpath

seed = 726
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

trainx = []
with open("data.train",'r') as f:
	for line in f:
		trainx.append([float(i) for i in line.split()[-256:]])
trainx = torch.FloatTensor(trainx)
trainx2 = trainx
class Autoencoder(nn.Module):
    def __init__(self, in_chs, hid_chs, out_chs):
        super(Autoencoder,self).__init__()
        
        U = (6 / (1 + hid_chs + 256)) ** 0.5
        self.encoder = nn.Linear(in_chs, hid_chs)
        self.decoder = nn.Linear(hid_chs, out_chs)
        self.encoder.weight.data.uniform_(-U,U)
        self.encoder.bias.data.uniform_(-U,U)
        self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        self.decoder.bias.data.uniform_(-U,U)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.tanh(x)
        x = self.decoder(x)
        return x

class Autoencoder2(nn.Module):
    def __init__(self, in_chs, hid_chs, out_chs):
        super(Autoencoder2,self).__init__()
        
        U = (6 / (1 + hid_chs + 256)) ** 0.5
        self.encoder = nn.Linear(in_chs, hid_chs)
        self.decoder = nn.Linear(hid_chs, out_chs)
        self.encoder.weight.data.uniform_(-U,U)
        self.encoder.bias.data.uniform_(-U,U)
        self.decoder.weight.data.uniform_(-U,U)
        self.decoder.bias.data.uniform_(-U,U)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.tanh(x)
        x = self.decoder(x)
        return x

def training(model, trainx):
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    for i in range(itr):
        print("epoch: ",i) 
        out = model(trainx.cuda())
        loss = criterion(trainx.cuda(), out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

itr = 5000
lr = 0.1

hid_l = [2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7]
log_hid_l = [1, 2, 3, 4, 5, 6, 7]

E_in = []
E_in2 = []
for i in range(7):
    model = Autoencoder(256,hid_l[i],256).cuda()
    model2 = Autoencoder2(256,hid_l[i],256).cuda()
    model = training(model, trainx)
    model2 = training(model2,trainx2)
    criterion = torch.nn.MSELoss()
    E_in.append(criterion(trainx.cuda() , model(trainx.cuda())).item())
    E_in2.append(criterion(trainx2.cuda() , model2(trainx2.cuda())).item())

plt.plot(log_hid_l , E_in2, marker = 'o', mec = 'red', mfc = 'red')
plt.plot(log_hid_l , E_in, marker = 'o', mec = 'red', mfc = 'red')
plt.legend(['p11', 'p13'], loc = 'best')
plt.xlabel(r'$log_{2}\tilde{d}$')
plt.ylabel(r'$E_{in}$')
plt.savefig('p13.jpg')
