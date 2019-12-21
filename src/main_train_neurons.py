import torch
from model import Encoder
import argparse
from dataset import Regression_dataset

parser = argparse.ArgumentParser()

parser.add_argument("--device",type = str,default="cuda")
parser.add_argument("state_dict_file",type = str)

args = parser.parse_args()

device = args.device

def state_to_fmap(state_dict_file):
    state_dict = torch.load(state_dict_file)
    model = Encoder().to(device)
    state = 0
    return state

pt_file = '/data3/valentin/dataset/images_fmri.pt'
reg_dataset = Regression_dataset(pt_file)