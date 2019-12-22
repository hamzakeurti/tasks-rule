import torch
from model import Encoder,RegressionDecoder
import argparse
from dataset import RegressionDataset
from torch.utils.data import random_split,DataLoader
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import torch.optim as optim
from utils import LOG_INFO


parser = argparse.ArgumentParser()

parser.add_argument("--task_name",type=str,help="Name of the task for file naming purposes")
parser.add_argument("--state_dict_file",type = str)
parser.add_argument("--batch_size",type=int,default=100)
parser.add_argument("--LOG_INTERVAL",type=int,default=50)
parser.add_argument("--EPOCHS",type=int,default=10)

parser.add_argument("--starting_layer",type=int,default=0)
parser.add_argument("--ending_layer",type=int,default=3)

parser.add_argument("--learning_rate",type=float,default=0.01)
parser.add_argument("--weight_decay",type=float,default=0.)

parser.add_argument("--device",type = str,default="cpu")

parser.add_argument("--ROI",type=str,default="All")
args = parser.parse_args()


device = args.device
pt_file = '/dev/shm/images_fmri.pt'

#['LHRSC', 'LHEarlyVis', 'RHEarlyVis', 'LHOPA', 'RHRSC', 'LHLOC', 'RHLOC', 'RHOPA', 'LHPPA', 'RHPPA', 'All']
ROIs = torch.load('/data3/valentin/datasets/ROIs.pt')
ROI_mask = np.ones(4438)
if args.ROI!="All":
    joined_ROI = [np.zeros(np.sum(ROIs[sub]['All']>=1)) for sub in range(3)]
    for ROI_name in ROIs[0].keys():
        if args.ROI in ROI_name:
            for sub in range(3):
                name = ROI_name
                if sub>0 and ROI_name[-2:]=='OC':
                    name = ROI_name[:-1]
                joined_ROI[sub] = joined_ROI[sub] + ROIs[sub][name][ROIs[sub]['All']>=1]

    ROI_mask = np.concatenate(joined_ROI)

encoder = Encoder()
encoder.load_state_dict(torch.load(args.state_dict_file))
encoder.eval()

reg_dataset = RegressionDataset(pt_file, encoder, device)

torch.manual_seed(0)
lengths = [int(len(reg_dataset)*0.8), len(reg_dataset)-int(len(reg_dataset)*0.8)]
train_dataset, test_dataset = random_split(reg_dataset, lengths)

train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)

dim_out = np.sum(ROI_mask>0)
reg_model = RegressionDecoder(args.starting_layer,args.ending_layer,dim_out).to(device)

optimizer = optim.Adam(reg_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)


def train(reg_model, device, train_loader, optimizer, epoch):
    reg_model.train()
    loss_list = []
    saved_loss_list = []
    for batch_idx, (fmap, target) in enumerate(train_loader):
        fmap = torch.cat([x.view(args.batch_size,-1) for x in fmap[args.starting_layer:args.ending_layer+1]],1).to(device)
        target = target.t()[ROI_mask>0].t().clone().detach().to(device).type(torch.float)
        optimizer.zero_grad()
        output = reg_model(fmap)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if batch_idx % args.LOG_INTERVAL == 0:
            loss_list_avg = np.sum(loss_list) / len(loss_list)
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}'.format(
                    epoch, batch_idx * len(fmap), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_list_avg)
            LOG_INFO(msg)
            saved_loss_list.append(loss_list_avg)
            loss_list.clear()
    return saved_loss_list


def test(reg_model, device, test_loader):
    reg_model.eval()
    test_loss = 0
    targets,outputs=[],[]
    size = 0
    with torch.no_grad():
        for batch_idx, (fmap, target) in enumerate(test_loader):
            fmap = torch.cat([x.view(args.batch_size,-1) for x in fmap[args.starting_layer:args.ending_layer+1]],1).to(device)
            target = target.t()[ROI_mask>0].t().clone().detach().to(device).type(torch.float)
            output = reg_model(fmap)
            test_loss+= F.mse_loss(output, target)
            targets.append(np.array(target.clone().detach().cpu()))
            outputs.append(np.array(output.clone().detach().cpu()))
            size += 1
    targets,outputs = np.concatenate((targets)),np.concatenate((outputs))
    EV = 1 - np.var(targets-outputs)/np.var(targets)
    test_loss /= size
    print('\nTest set: Average loss: {:.4f}, EV= {:.4f}\n'.format(
        test_loss,EV))
    return test_loss


if __name__=='__main__':
    min_loss = 5000
    print(reg_model)
    train_losses,train_accs,test_losses,test_accs = [],[],[],[]
    for epoch in range(1, args.EPOCHS + 1):
        train_loss= train(reg_model, device, train_loader, optimizer, epoch)
        test_loss = test(reg_model, device, test_loader)
        train_losses.extend(train_loss)
        test_losses.append(test_loss)
        if test_loss<min_loss:
            min_loss = test_loss
            torch.save(reg_model.state_dict(),"saved_models/"+args.task_name+"_reg_model.pth")