import pickle
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from numpy import asarray
import numpy as np
import torch
import cv2

def pickle_to_dataset(pickle_file):
    data = open(pickle_file,"rb")
    d = pickle.load(data)
    train,test = d
    return train,test

def ID_to_picture(image_id, task_idx):
    if task_idx == 2:
        image_path = "/data4/chenhaoran/{}".format(image_id)
    else:
        image_path = "/data4/chenhaoran/mscoco/{}2014/COCO_{}2014_{:012d}.jpg".format('train','train',image_id)
        img = cv2.imread(image_path)
    res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)/255
    return res

class Classification_dataset(Dataset):
    def __init__(self,pickle_file, task_idx, mode="train"):
        train,test = pickle_to_dataset(pickle_file)
        self.data = test[0]
        self.labels = test[1]
        if mode =="train":
            self.data = train[0]
            self.labels = train[1]

        self.mode=mode
        self.task_idx = task_idx

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        image = ID_to_picture(self.data[idx], self.task_idx)
        return image,self.labels[idx]

class ReconstructionDataset(Dataset):
    def __init__(self,pickle_file,mode="train"):
        train,test = pickle_to_dataset(pickle_file)
        if mode =="train":
            self.data = train[0]
        else:
            self.data = test[0]
        self.mode=mode
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        image = ID_to_picture(self.data[idx])
        return image

class Regression_dataset(Dataset):
    def __init__(self,pt_file):
        self.data = torch.load(pt_file)
        #Data is a list of [image_name,image_array,fMRI_array]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        return self.data[idx]
