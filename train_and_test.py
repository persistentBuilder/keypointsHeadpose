from __future__ import print_function, division
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from biwiDataLoader import biwiDataset
from simpleNet import simpNet
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random      
import time



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device,use_cuda)
#device = torch.device("cuda" if use_cuda else "cpu")


kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
BATCH_SIZE = 128
NB_EPOCHS = 1020
fer = 1000*np.ones((3))
path_and_ground_truth_file = 'biwiGT'


def mse_loss(input, target):
    return torch.sum((input - target) ** 2)

def train(epochs, it):
    model.train()
    for epoch in range(0,epochs):
        train_loss = 0
        t = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()   # clear gradients for next train
            output = model(data)
            loss = loss_func(output, target)  
            train_loss += loss*output.shape[0]*3   # must be (1. nn output, 2. target)
            loss.backward() 
            optimizer.step()
        train_loss = train_loss/(len(train_loader.dataset)*3) 
        elapsed = time.time() - t
        print (' -%d-  Epoch [%d/%d]'%(elapsed, epoch+1, NB_EPOCHS))
        print ('Training samples: %d Train Loss: %.5f'%(len(train_loader.dataset), train_loss.item()))
        test(prnt=False, it=it, epoch=epoch)
        #if batch_idx % 10 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))

def test(prnt=False, it=-1, epoch=-1):
    model.eval()
    test_loss = 0
    correct = 0
    yer = 0
    per = 0
    rer = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)     # must be (1. nn output, 2. target)
            test_loss += loss*3*output.shape[0]
            for i in range(output.shape[0]):
                    yer += abs(output[i][0]-target[i][0])
                    per += abs(output[i][1]-target[i][1])
                    rer += abs(output[i][2]-target[i][2])
    test_loss /= (len(test_loader.dataset)*3)
    print ('Test samples: %d Test Loss: %.5f'%(len(test_loader.dataset), test_loss.item()))
    er1 = ((yer.item())/len(test_loader.dataset))*180
    er2 = ((per.item())/len(test_loader.dataset))*180
    er3 = ((rer.item())/len(test_loader.dataset))*180
    if er1+er2+er3 < fer[0] + fer[1] + fer[2]:
        print(epoch, "improved", (er1+er2+er3)/3)
        fer[0] = er1
        fer[1] = er2
        fer[2] = er3
    if prnt:
        print ('Mean Absolute Error: Yaw %.5f, Pitch %.5f, Roll %.5f, Avg %.5f'%(er1, er2, er3, (er1+er2+er3)/3))

als = []
for num in range(1, 25):
    als.append(num)
als = set(als) 
selected_test_set = [12, 16, 17]    
train_set = als-set(selected_test_set)
train_set = list(train_set)
print(train_set, selected_test_set)

train_dataset = biwiDataset(path_and_ground_truth_file, 1)
train_dataset.select_sets(sets=train_set)
#for line in train_dataset.lines:
#    print(line)

print("\n==============================\n")
test_dataset = biwiDataset(path_and_ground_truth_file, 0)
test_dataset.select_sets(sets=selected_test_set)
#for line in test_dataset.lines:
#    print(line)


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

model = simpNet().to(device)
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
#    torch.backends.cudnn.benchmark=True    #model = bmvcNet().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.000001)
#optimizer = optim.Adadelta(model.parameters())
loss_func = torch.nn.MSELoss()
train(NB_EPOCHS, it=1)
test(prnt=True, it=1)

print(' MAE: Yaw %.5f, Pitch %.5f, Roll %.5f, Avg %.5f'%(fer[0], fer[1], fer[2], (fer[0]+fer[1]+fer[2])/3))
