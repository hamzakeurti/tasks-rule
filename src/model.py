import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(3,64,7,padding = 7//2,stride = 2)
        self.conv2 = nn.Conv2d(64,128,3,padding = 3//2)
        self.conv3 = nn.Conv2d(128,256,3,padding = 3//2)
        self.conv4 = nn.Conv2d(256,512,3,padding = 3//2)

        self.norm1=nn.BatchNorm2d(64)
        self.norm2=nn.BatchNorm2d(128)
        self.norm3=nn.BatchNorm2d(256)
        self.norm4=nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(3,padding = 3//2,stride = 2)
        self.relu = nn.ReLU()
        self.pool_final = nn.MaxPool2d(7)
        self.downsample=nn.AvgPool2d(8,stride=2,padding=3)
    def forward(self,x):
        x = self.pool(self.relu(self.norm1(self.conv1(x))))
        V1 = x

        x = self.pool(self.relu(self.norm2(self.conv2(x))))
        V2 = x

        x = self.pool(self.relu(self.norm3(self.conv3(x))))
        V4 = x

        x = self.pool(self.relu(self.norm4(self.conv4(x))))
        IT = x
        
        x = self.pool_final(x)
        return self.downsample(V1),self.downsample(V2),V4,IT,x

class Classification_Decoder(nn.Module):
    # Same model for single class and multi-class classification, just change the loss function
    def __init__(self,in_dim, out_dim):
        super(Classification_Decoder,self).__init__()
        self.linear = nn.Linear(in_dim,out_dim)
    def forward(self,x):
        x = self.linear(x)
        return x
    

class ReconstructionDecoder(nn.Module):
    def __init__(self,in_features,image_shape):
        super(ReconstructionDecoder,self).__init__()
        self.linear = nn.Linear(in_features=in_features,out_features=np.prod(image_shape))
        self.ouput_shape = image_shape
        
    def forward(self,code):
        x_hat = self.linear(code)
        x_hat = x_hat.view((-1,)+self.ouput_shape)
        return x_hat

class RegressionDecoder(nn.Module):
    def __init__(self,starting_layer,ending_layer,dim_out=4438):
        super(RegressionDecoder,self).__init__()
        sizes = [64*28*28,128*14*14,256*14*14,512*7*7,512]
        self.linear1 = nn.Linear(in_features = sum(sizes[starting_layer:ending_layer+1]), out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=dim_out)
        
    def forward(self, fmap):
        out = self.linear1(fmap)
        out = self.linear2(out)
        return out