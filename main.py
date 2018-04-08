import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import Net
import glob
import re
import numpy as np
import h5py

cuda = False

model = Net()
if cuda:
    model.cuda()

criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()
#criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)
#optimizer = optim.SGD(model.parameters(), lr=0.1)

def train_loader(batch_size=20, epochs=10):
    with h5py.File("training_data.h5", "r") as f:
        cats = f["cats"].value
        patches = f["patches"].value
        
    n_samples = patches.shape[0]
    batches = n_samples // batch_size * epochs
    print("attempting %d batches, %d samples per batch" % (batches, batch_size))

    for i in range(batches):
        print(i)
        idxs = np.random.choice(np.arange(n_samples), batch_size)
        
        bcats = cats[idxs]
        bpatches = patches[idxs]

        s = bpatches.shape
        bpatches = bpatches.reshape(s[0], 1, s[1])

        yield bcats, bpatches

    
def train(cuda, save_path):
    for cats, patches in train_loader():
        border = (cats.shape[2] - model.S) // 2

        # just threshold for now
        cats = cats[:,0,border:-border]
        #print(cats.shape)
        
        data = torch.Tensor(patches)
        target = torch.Tensor(cats)
        
        if cuda:
            data.cuda()
            target.cuda()

        data, target = Variable(data), Variable(target)

        if cuda:
            target.cuda()
            
        optimizer.zero_grad()
        
        output = model(data)
        #print(output.size(), target.size())
        #print(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #print(model.conv1.weight.data.numpy()[0])

        print('Loss: %0.7f' % loss.data)
        torch.save(model.state_dict(), save_path)
        #del data, target, output

train(cuda=cuda, save_path='model_weights.torch')
