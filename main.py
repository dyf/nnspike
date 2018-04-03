import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import Net

net = Net()
inpt = torch.randn(1,2000)
inpt = Variable(inpt.unsqueeze(0))
output = net(inpt)
target = Variable(torch.LongTensor([3]))
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
loss.backward()
#o = m(x)
#print(o.size())

