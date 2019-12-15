import pickle
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from numpy import asarray


def pickle_to_dataset(pickle):
    data = open(pickle,"rb")
    d = pickle.load(data)
    train,test = d
    return train,test

def ID_to_picture(image_id,mode = "train"):
    image_path = "./%s2014/COCO_%s2014_%012d.jpg" % (mode,mode,image_id)
    image = Image.open(image_path)
    image = asarray(image)
    image = image.resize((224,224))
    return image


class Classification_dataset(Dataset):
    def __init__(self,pickle,mode="train"):
        train,test = pickle_to_dataset(pickle)
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