import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import LOG_INFO
from model import Encoder,Classification_Decoder
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("task_name",type=str,help="Name of the task for file naming purposes")
parser.add_argument("dataset",type=str,help = "pickle file associated to the dataset")

parser.add_argument("device",type = str,default="cuda")

parser.add_argument("batch_size",type=int,help="Batch size",default=100)
parser.add_argument("LOG_INTERVAL",type=int,default=50)
parser.add_argument("EPOCHS",type=int,default=50)


parser.add_argument("momentum",type = float,default=0.)
parser.add_argument("learning_rate",type=float,default=0.01)
parser.add_argument("weight_decay",type=float,default=0.)
parser.add_argument("Encoder_out",type=int,default=512)
parser.add_argument("Decoder_out",type=int,default=10)

args = parser.parse_args()



device = args.device
kwargs = {'num_workers': 1, 'pin_memory': True}


encoder = Encoder()
decoder = Classification_Decoder(args.Encoder_out,args.Decoder_out).to(device)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

def train(encoder,decoder, device, train_loader, optimizer, epoch):
    encoder.train()
    decoder.train()
    loss_list = []
    acc_list = []
    saved_loss_list = []
    saved_acc_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = encoder(data)
        output = output.view(-1,args.Encoder_out)
        output = decoder(output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).float().mean()
        acc_list.append(acc.item())
        if batch_idx % args.LOG_INTERVAL == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(loss_list), np.mean(acc_list))
            LOG_INFO(msg)
            saved_loss_list.append(np.mean(loss_list))
            saved_acc_list.append(np.mean(acc_list))
            loss_list.clear()
            acc_list.clear()
    return saved_loss_list,saved_acc_list


def test(encoder,decoder, device, test_data,test_label):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data, target = test_data.to(device), test_label.to(device)
        data = data.view(data.shape[0],1,28,28)
        output = encoder(data)
        output = output.view(-1,args.Encoder_out)
        output = decoder(output)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))
    return test_loss,correct/len(test_data)

if __name__=='__main__':
    min_loss = 5000
    train_losses,train_accs,test_losses,test_accs = [],[],[],[]
    for epoch in range(1, args.EPOCHS + 1):
        train_loss, train_acc = train(encoder,decoder, device, train_loader, optimizer, epoch)
        test_loss,test_acc = test(encoder,decoder, device, test_data,test_label)
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if test_loss<min_loss:
            min_loss = test_loss
            best_model = deepcopy(encoder)
    prefix = args.task_name
    np.save(prefix + 'test_losses.npy',test_losses)
    np.save(prefix + 'train_losses.npy',train_losses)
    np.save(prefix + 'test_accs.npy',test_accs)
    np.save(prefix + 'train_accs.npy',train_accs)