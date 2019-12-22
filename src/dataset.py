import pickle
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from numpy import asarray
import torch
import cv2

def pickle_to_dataset(pickle_file):
    data = open(pickle_file,"rb")
    d = pickle.load(data)
    train,test = d
    return train,test

def ID_to_picture(image_id, task_idx=1):
    if task_idx in {2, 4, 5}:
        image_path = "/dev/shm/{}".format(image_id)
    else:
        image_path = "/dev/shm/mscoco/{}2014/COCO_{}2014_{:012d}.jpg".format('train','train',image_id)
    img = cv2.imread(image_path)
    if type(img)!=np.ndarray:
        image_path = "/dev/shm/mscoco/{}2014/COCO_{}2014_{:012d}.jpg".format('val','val',image_id)
        img = cv2.imread(image_path)
    res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
    res = np.swapaxes(res,0,2)
    res = np.swapaxes(res,1,2)
    res = np.ndarray.astype(res,np.float32)/255
    return res

class ClassificationDataset(Dataset):
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

class RegressionDataset(Dataset):
    def __init__(self,pt_file,encoder):
        self.data = torch.load(pt_file)
        #Data is a list of [image_name,image_array,fMRI_array]
        self.encoder = encoder

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img = torch.tensor(self.data[idx][1]).unsqueeze(0)
        return self.encoder(img),self.data[idx][2]