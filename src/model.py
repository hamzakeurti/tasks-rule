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

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        V1 = x

        x = self.pool(self.relu(self.conv2(x)))
        V2 = x

        x = self.pool(self.relu(self.conv3(x)))
        V4 = x

        x = self.pool(self.relu(self.conv4(x)))
        IT = x

        return V1,V2,V4,IT
        