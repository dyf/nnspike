import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.N = 30
        self.k = 25
        self.p = self.N * 1880
        self.cats = 4
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.N, kernel_size=self.k)
        self.conv2 = nn.Conv1d(in_channels=self.N, out_channels=self.N, kernel_size=self.k)
        self.conv3 = nn.Conv1d(in_channels=self.N, out_channels=self.N, kernel_size=self.k)
        self.conv4 = nn.Conv1d(in_channels=self.N, out_channels=self.N, kernel_size=self.k)
        self.conv5 = nn.Conv1d(in_channels=self.N, out_channels=self.N, kernel_size=self.k)
        self.fc = nn.Linear(self.p, self.cats)

    def forward(self, x):
        #x = F.relu(F.max_pool1d(self.conv1(x), 2))
        #x = F.relu(F.max_pool1d(self.conv2(x), 2))
        #x = F.relu(F.max_pool1d(self.conv2(x), 2))
        #x = F.relu(F.max_pool1d(self.conv2(x), 2))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        print(x.size())
        x = x.view(-1,self.p)
        x = self.fc(x)
        #x = F.log_softmax(x, dim=-1)
        return x
