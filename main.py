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

criterion = nn.MSELoss()#nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())#, lr=0.001)

def train_loader(batch_size=100):
    with h5py.File("training_data.h5", "r") as f:
        cats = f["cats"].value
        patches = f["patches"].value
        
    n_samples = patches.shape[0]
    batches = n_samples // batch_size
    print("attempting %d batches, %d samples per batch" % (batches, batch_size))

    for i in range(batches):
        print(i)
        idxs = np.random.choice(np.arange(n_samples), batch_size)
        
        bcats = cats[idxs]
        bpatches = patches[idxs]

        s = bpatches.shape
        bpatches = bpatches.reshape(s[0], 1, s[1])
        bcats = bcats.reshape(s[0], 1, s[1])

        yield bcats, bpatches

    
def train(cuda, save_path):
    for cats, patches in train_loader():
        data = torch.Tensor(patches)
        
        if cuda:
            data.cuda()

        data = Variable(data)            
        output = model(data)
        
        border = (cats.shape[2] - output.size()[2]) // 2
        cats = cats[:,:,border:border+output.size()[2]]
        target = torch.Tensor(cats)

        if cuda:
            target.cuda()
            
        target = Variable(target)
        
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Loss: %0.7f' % loss.data)
        torch.save(model.state_dict(), save_path)
        del data, target, output

train(cuda=cuda, save_path='model_weights.torch')
