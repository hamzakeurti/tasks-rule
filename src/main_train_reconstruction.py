import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from utils import LOG_INFO
from model import Encoder,ReconstructionDecoder
from copy import deepcopy
from dataset import ReconstructionDataset
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--task_name",type=str,help="Name of the task for file naming purposes")
parser.add_argument("--dataset",type=str,help = "pickle file associated to the dataset",default = "data/single-class-classification-dataset.pkl")

parser.add_argument("--device",type = str,default="cuda")

parser.add_argument("--batch_size",type=int,help="Batch size",default=100)
parser.add_argument("--LOG_INTERVAL",type=int,default=50)
parser.add_argument("--EPOCHS",type=int,default=50)


parser.add_argument("--momentum",type = float,default=0.)
parser.add_argument("--learning_rate",type=float,default=0.01)
parser.add_argument("--weight_decay",type=float,default=0.)
parser.add_argument("--Encoder_out",type=int,default=512)
parser.add_argument("--image_shape",type=list,default=(3,224,224))

parser.add_argument("--multitask_mode",type=int,default=0)

args = parser.parse_args()

train_dataset = ReconstructionDataset(args.dataset)
test_dataset = ReconstructionDataset(args.dataset,mode="val")

train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

device = args.device
kwargs = {'num_workers': 1, 'pin_memory': True}


encoder = Encoder().to(device)
decoder = ReconstructionDecoder(args.Encoder_out,args.image_shape).to(device)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

def train(encoder,decoder, device, train_loader, optimizer, epoch, max_iterations = None):
    encoder.train()
    decoder.train()
    loss_list = []
    saved_loss_list = []
    num_iter = max_iterations if max_iterations else len(train_loader)
    for batch_idx, x in enumerate(train_loader):
        if batch_idx == max_iterations:
            break
        x = x.to(device)
        target = x.clone().detach().to(device)
        optimizer.zero_grad()
        output = encoder(x)
        output = output[4].view(-1,args.Encoder_out)
        output = decoder(output)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if batch_idx % args.LOG_INTERVAL == 0:
            msg = f'Train Epoch: {epoch} [{batch_idx * len(x)}/{len(train_loader.dataset)} ({100. * batch_idx / num_iter:.0f}%)]\tAvg Loss: {np.mean(loss_list):.4f}'
            LOG_INFO(msg)
            saved_loss_list.append(np.mean(loss_list))
            loss_list.clear()
    return saved_loss_list


def test(encoder,decoder, device, test_loader, max_iterations = None):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, x in enumerate(test_loader):
            if batch_idx == max_iterations:
                break
            x = x.to(device)
            target = x.clone().detach().to(device)
            optimizer.zero_grad()
            output = encoder(x)
            output = output[4].view(-1,args.Encoder_out)
            output = decoder(output)
            test_loss+= F.mse_loss(output, target)
    if max_iterations:
        size = max_iterations * args.batch_size
    else:
        size = len(test_loader) * args.batch_size
    test_loss /= size
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')
    return test_loss

if __name__=='__main__':
    save_subdir = f"saved_models/{args.task_name}-plots/"
    if not os.path.exists(save_subdir):
        os.mkdir(save_subdir)
    min_loss = 5000
    train_losses,test_losses = [],[]
    for epoch in range(1, args.EPOCHS + 1):
        train_loss = train(encoder,decoder, device, train_loader, optimizer, epoch)
        test_loss = test(encoder,decoder, device, test_loader)
        train_losses.extend(train_loss)
        test_losses.append(test_loss)
        if test_loss<min_loss:
            min_loss = test_loss
            best_model = deepcopy(encoder)
            best_decoder = deepcopy(decoder)
        # Plot
        for batch_idx, x in enumerate(test_loader):
            x = x.to(device)
            best_model.eval()
            best_decoder.eval()
            out = best_model(x)
            out = out[4].view(-1,args.Encoder_out)
            out = best_decoder(out)
            fig , axes = plt.subplots(10,2,figsize=(10,40))
            for i in range(10):
                axes[i][0].imshow(np.moveaxis(x.detach().numpy()[i],0,-1))
                axes[i][0].tick_params(labelbottom=False,labelleft=False,left=False,bottom=False)
                axes[i][0].set_title('Original')
                axes[i][1].imshow(np.moveaxis(out.detach().numpy()[i],0,-1))
                axes[i][1].tick_params(labelbottom=False,labelleft=False,left=False,bottom=False)
                axes[i][1].set_title('Reconstructed')
            fig.savefig(f'{save_subdir}{epoch}.png')
            fig.close()
            break
    prefix = args.task_name
    np.save(prefix + '_test_losses.npy',test_losses)
    np.save(prefix + '_train_losses.npy',train_losses)
    torch.save(best_model.state_dict(),"saved_models/"+args.task_name+".pth")