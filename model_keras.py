import keras

class Net(nn.Module):
    N = 100
    K = 25
    CATS = 3
    S = 152
    
    def __init__(self):
        super(Net, self).__init__()
        
        #self.conv1 = nn.Conv1d(in_channels=1,                out_channels=self.CATS,        kernel_size=self.K, groups=1)
        #self.conv2 = nn.Conv1d(in_channels=self.CATS,        out_channels=self.N*self.CATS, kernel_size=self.K, groups=self.CATS)
        #self.conv3 = nn.Conv1d(in_channels=self.N*self.CATS, out_channels=self.N*self.CATS, kernel_size=self.K, groups=self.CATS)
        #self.conv4 = nn.Conv1d(in_channels=self.N*self.CATS, out_channels=self.N*self.CATS, kernel_size=self.K, groups=self.CATS)
        #self.conv5 = nn.Conv1d(in_channels=self.N*self.CATS, out_channels=self.CATS,        kernel_size=self.K, groups=self.CATS)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.N, kernel_size=self.K)
        self.conv2 = nn.Conv1d(in_channels=self.N, out_channels=self.N, kernel_size=self.K)
        self.fc = nn.Linear(self.N*self.S, self.S)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        #x = F.relu(self.conv5(x))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.S*self.N)
        x = self.fc(x)
        x = F.sigmoid(x)
        #x = F.log_softmax(self.fc(x), dim=-1)
        

        #x = F.sigmoid(x)
        #print(x.size())
        #x = x.view(-1,self.p)
        #x = self.sig(self.fc(x))
        #x = F.log_softmax(x, dim=-1)
        return x
