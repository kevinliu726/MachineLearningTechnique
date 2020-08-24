import os
import sys
import class_model
import torch
from torch.utils.data import DataLoader
import random
import numpy as np


def test(model, test_dataset):
    print("test!")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    ans_list = []
    
    model.eval() # model evaluation: dropout, etc...
    with torch.no_grad(): # set all the requires_grad flag to false
        for image in test_loader: # one batch
            y_predict = model(image.cuda()) 
            predicted = torch.argmax(y_predict.cuda(), dim = 1)

            for y in predicted:
                ans_list.append(y)

    return ans_list

# python3 test.py [test dir] [.pth] [.csv]
def main():
    test_dpath = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]

    test_dataset = class_model.Dataset(test_dpath, mode="test")

    model = class_model.kCNN().cuda()
    model.load_state_dict(torch.load(model_path))

    #test
    ans_list = test(model, test_dataset)

    #submit.csv
    with open(output_path, 'w') as f:
        f.write('image_id,label\n')
        for i in range(len(ans_list)):
            f.write('{},{}\n'.format(test_dataset.filename_dict[i], chr(ans_list[i]+ord('A'))))
    
    return 

if __name__ == "__main__":
    seed = 726
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
