import os
import sys
import class_model
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
import random
import numpy as np


def _validation(model, valid_dataset):
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    loss_function = nn.CrossEntropyLoss() #sum cross entropy over one batch
    model.eval()
    with torch.no_grad():
        correct = 0.0
        val_loss = 0.0
        for image, label in valid_loader: #one batch
            y_predict = model(image.cuda()) 
            predicted = torch.argmax(y_predict.cpu(), dim = 1)

            val_loss += loss_function(y_predict, label.cuda()).item()
            correct += (predicted == label).sum().item()
    return correct, val_loss


def _train(model, train_dataset, valid_dataset):
    print("Loading training data...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    loss_function = nn.CrossEntropyLoss() #sum cross entropy over one batch

    print("train!")
    for epoch in range(num_epochs):
        train_acc = 0.0
        train_loss = 0.0
        model.train()
        if epoch == 56:
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
        for image, label in train_loader: # one batch
            #print("image.size():", image.size())
            optimizer.zero_grad()
            y_predict = model(image.cuda()) # probabilities of 11 classes

            # Backprop and perform Adam optimisation
            loss = loss_function(y_predict, label.cuda())
            loss.backward()
            optimizer.step()
            #loss and accuracy
            predicted = torch.argmax(y_predict.cpu(), dim=1)
            train_acc += (predicted == label).sum().item()
            train_loss += loss.item()

        if valid_dataset == None:
            print("epoch: %2d, train accuracy: %3.5f" % (epoch, train_acc/ train_dataset.__len__()))
        else:
            valid_acc, valid_loss = _validation(model, valid_dataset)
            print("epoch: %2d, train accuracy: %3.5f, train loss: %3.5f, valid accuracy: %3.5f, valid loss: %3.5f" % (epoch, train_acc/ train_dataset.__len__(), train_loss / train_dataset.__len__(),valid_acc/ valid_dataset.__len__(), valid_loss / valid_dataset.__len__()))

    return model


# python3 train.py [root] [.pth]
def main():
    train_dpath = os.path.join(sys.argv[1], "train") # no / ending
    valid_dpath = os.path.join(sys.argv[1], "validation") # no / endin
    model_fpath = sys.argv[2]

    train_dataset = class_model.Dataset(train_dpath, mode="train")
    valid_dataset = class_model.Dataset(valid_dpath, mode="validation")

    model = class_model.kCNN().cuda()
    model = _train(model, train_dataset, valid_dataset)

    # fine tune with validation
    train_valid_dataset = ConcatDataset([train_dataset, valid_dataset])
    model = _train(model, train_valid_dataset, valid_dataset=None)

    #save model
    torch.save(model.state_dict(), model_fpath)

    return




if __name__ == "__main__":
    seed = 111
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # arguments
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 80
    weight_decay = 5e-6

    main()
