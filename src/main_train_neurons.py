import torch
from model import Encoder
import argparse
from dataset import RegressionDataset

parser = argparse.ArgumentParser()

parser.add_argument("--state_dict_file",type = str)

args = parser.parse_args()



pt_file = '/data3/valentdin/dataset/images_fmri.pt'


encoder = Encoder()
encoder.load_state_dict(torch.load(args.state_dict_file))


reg_dataset = RegressionDataset(pt_file,encoder)