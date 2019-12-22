import torch
from model import Encoder
import argparse
from dataset import RegressionDataset
from torch.utils.data import random_split,DataLoader

parser = argparse.ArgumentParser()

parser.add_argument("--state_dict_file",type = str)
parser.add_argument("--batch_size",type=int,default=50)
args = parser.parse_args()



pt_file = '/data3/valentin/datasets/images_fmri.pt'


encoder = Encoder()
encoder.load_state_dict(torch.load(args.state_dict_file))
encoder.eval()

reg_dataset = RegressionDataset(pt_file,encoder)

torch.manual_seed(0)
lengths = [int(len(reg_dataset)*0.8), int(len(reg_dataset)*0.2)]
train_dataset, test_dataset = random_split(reg_dataset, lengths)

train_dataloader,test_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True),DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)
