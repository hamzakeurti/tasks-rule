import pickle
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from numpy import asarray
import numpy as np

def pickle_to_dataset(pickle_file):
    data = open(pickle_file,"rb")
    d = pickle.load(data)
    train,test = d
    return train,test

def ID_to_picture(image_id):
    image_path = "/data4/chenhaoran/mscoco/train2014/COCO_train2014_%012d.jpg" % (image_id)
    image = Image.open(image_path)
    image = asarray(image)
    image = np.resize(image,(224,224,3))
    image=np.swapaxes(image,0,2)
    image = np.swapaxes(image,1,2)
    image = np.ndarray.astype(image,np.float32)/255
    return image


class Classification_dataset(Dataset):
    def __init__(self,pickle_file,mode="train"):
        train,test = pickle_to_dataset(pickle_file)
        self.data = test[0]
        self.labels = test[1]
        if mode =="train":
            self.data = train[0]
            self.labels = train[1]
            
        self.mode=mode
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        image = ID_to_picture(self.data[idx],self.mode)
        return image,self.labels[idx]