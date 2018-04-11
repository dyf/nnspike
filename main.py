import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
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
#criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
#criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)
#optimizer = optim.SGD(model.parameters(), lr=0.1)

def train_loader(batch_size=1, epochs=30):
    with h5py.File("training_data.h5", "r") as f:
        cats = f["cats"].value
        patches = f["patches"].value
        
    n_samples = patches.shape[0]
    batches = n_samples // batch_size * epochs
    print("attempting %d batches, %d samples per batch" % (batches, batch_size))

    for i in range(batches):
        idxs = np.random.choice(np.arange(n_samples), batch_size)
        
        bcats = cats[idxs]
        bpatches = patches[idxs]

        s = bpatches.shape
        bpatches = bpatches.reshape(s[0], 1, s[1])

        yield bcats, bpatches

    
def train(cuda, save_dir):
    for i, (cats, patches) in enumerate(train_loader()):
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

        
        if (i % 100) == 0:
            print('Batch %d, Loss: %0.7f' % (i+1, loss.data))
            #print(model.conv1.weight.data.numpy()[0])
            d =   model.conv1.weight.data.numpy()

            fig, (ax1,ax2) = plt.subplots(2,1)
            for row in range(d.shape[0]):
                ax1.plot(d[row,0])

            ax2.plot(data.data.numpy()[0,0,border:-border], label='v')
            ax2.plot(target.data.numpy()[0], label='target')
            ax2.plot(F.softmax(output).data.numpy()[0], label='output')
            ax2.legend()
            
            plt.suptitle('Loss: %0.7f' % loss.data)
            plt.savefig(os.path.join(save_dir, "%03d.png" % i))
            plt.close()



            torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.torch'))
        del data, target, output

train(cuda=cuda, save_dir='/mnt/c/Users/davidf/workspace/nnspike')
