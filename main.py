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

cuda = False

model = Net()
if cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train_loader():
    pat = re.compile('.*?_(\d+).npy')
    for f in glob.glob('patches/patches*'):
        m = re.match(pat, f)
        if m:
            print(f)
            pid = int(m.group(1))
            pfile = 'patches/patches_%04d.npy' % pid
            cfile = 'patches/cats_%04d.npy' % pid

            patches = np.load(pfile)
            cats = np.load(cfile)
        
            s = patches.shape
            patches = patches.reshape(s[0], 1, s[1])

            yield cats, patches

    
def train(cuda, save_path):
    for cats, patches in train_loader():
        data = torch.Tensor(patches)
        target = torch.LongTensor(cats)

        if cuda:
            data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print('Loss: %.6f' % loss.data[0])
        torch.save(model.state_dict(), save_path)
        del data, target, output

train(cuda=cuda, save_path='model_weights.torch')
