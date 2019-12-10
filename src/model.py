import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(3,64,7,padding = 7//2,stride = 2)
        self.conv2 = nn.Conv2d(64,128,3,padding = 3//2)
        self.conv3 = nn.Conv2d(128,256,3,padding = 3//2)
        self.conv4 = nn.Conv2d(256,512,3,padding = 3//2)

        self.pool = nn.MaxPool2d(3,padding = 3//2,stride = 2)
        self.relu = nn.ReLU()
        self.pool_final = nn.MaxPool2d(7)
    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        V1 = x

        x = self.pool(self.relu(self.conv2(x)))
        V2 = x

        x = self.pool(self.relu(self.conv3(x)))
        V4 = x

        x = self.pool(self.relu(self.conv4(x)))
        IT = x
        
        x = self.pool_final(x)
        return V1,V2,V4,IT,x

class Classification_Decoder(nn.Module):
    # Same model for single class and multi-class classification, just change the loss function
    def __init__(self,in_dim, out_dim):
        super(Classification_Decoder,self).__init__()
        self.linear = nn.Linear(in_dim,out_dim)
    def forward(self,x):
        x = self.linear(x)
        return x