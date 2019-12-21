import torch
from model import Encoder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--device",type = str,default="cuda")
parser.add_argument("state_dict_file",type = str)

args = parser.parse_args()

device = args.device

stimuli = 0

def state_to_fmap(state_dict_file):
    state_dict = torch.load(state_dict_file)
    model = Encoder().to(device)
    return model(stimuli)

