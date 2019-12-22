import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from utils import LOG_INFO
from model import Encoder,Classification_Decoder
from copy import deepcopy
from dataset import ClassificationDataset
import argparse
from sklearn.metrics import average_precision_score


parser = argparse.ArgumentParser()
parser.add_argument("--task_name",type=str,help="Name of the task for file naming purposes")
parser.add_argument("--task_index",type=int,help="Index of the task")
parser.add_argument("--dataset",type=str,help="pickle file associated to the dataset",default="data/single-class-classification-dataset.pkl")

parser.add_argument("--device",type = str,default="cuda")

parser.add_argument("--batch_size",type=int,help="Batch size",default=100)
parser.add_argument("--LOG_INTERVAL",type=int,default=50)
parser.add_argument("--EPOCHS",type=int,default=50)


parser.add_argument("--learning_rate",type=float,default=0.01)
parser.add_argument("--weight_decay",type=float,default=0.)
parser.add_argument("--Encoder_out",type=int,default=512)
parser.add_argument("--Decoder_out",type=int,default=10)


args = parser.parse_args()

train_dataset = ClassificationDataset(args.dataset, args.task_index)
test_dataset = ClassificationDataset(args.dataset, args.task_index, mode="val")

train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True)

device = args.device
kwargs = {'num_workers': 1, 'pin_memory': True}


encoder = Encoder().to(device)
decoder = Classification_Decoder(args.Encoder_out,args.Decoder_out).to(device)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)

def train(encoder,decoder, device, train_loader, optimizer, epoch):
    encoder.train()
    decoder.train()
    loss_list = []
    acc_list = []
    outputs, targets = [], []
    saved_loss_list = []
    saved_acc_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.clone().detach().to(device)
        if args.task_index == 3:
                target = target.type(torch.float)
        optimizer.zero_grad()
        output = encoder(data)
        output = output[4].view(-1,args.Encoder_out)
        output = decoder(output)
        if args.task_index in [1, 2, 4, 5]:
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            acc = pred.eq(target.view_as(pred)).float().mean()
            acc_list.append(acc.item())
        else:
            loss = F.binary_cross_entropy_with_logits(output,target)
            outputs.append(output.cpu().detach().numpy())
            targets.append(target.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if batch_idx % args.LOG_INTERVAL == 0:
            if args.task_index in [1, 2, 4, 5]:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(loss_list), np.mean(acc_list))
                LOG_INFO(msg)
                saved_loss_list.append(np.mean(loss_list))
                saved_acc_list.append(np.mean(acc_list))
                loss_list.clear()
                acc_list.clear()
            else:
                y_true = np.concatenate(targets, axis=0)
                y_score = np.concatenate(outputs, axis=0)
                average_precision = average_precision_score(y_true, y_score)
                saved_loss_list.append(np.mean(loss_list))
                saved_acc_list.append(average_precision)
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Precision: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), saved_loss_list[-1], average_precision)
                LOG_INFO(msg)

                loss_list.clear()
                targets.clear()
                outputs.clear()

    return saved_loss_list,saved_acc_list


def test(encoder,decoder, device, test_loader):
    encoder.eval()
    decoder.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), torch.tensor(target).to(device)
            if args.task_index == 2:
                target = target.type(torch.float)
            output = encoder(data)
            output = output[4].view(-1,args.Encoder_out)
            output = decoder(output)
            if args.task_index in [1, 2, 4, 5]:
                test_loss+= F.cross_entropy(output, target)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
            else:
                test_loss+= F.binary_cross_entropy_with_logits(output,target)
    size = len(test_loader) * args.batch_size
    test_loss /= size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, size,
        100. * correct / size))
    return test_loss,correct/size

if __name__=='__main__':
    min_loss = 5000
    train_losses,train_accs,test_losses,test_accs = [],[],[],[]
    for epoch in range(1, args.EPOCHS + 1):
        train_loss, train_acc = train(encoder,decoder, device, train_loader, optimizer, epoch)
        test_loss,test_acc = test(encoder,decoder, device, test_loader)
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if test_loss<min_loss:
            min_loss = test_loss
            best_model = deepcopy(encoder)
    prefix = args.task_name
    np.save(prefix + '_test_losses.npy',test_losses)
    np.save(prefix + '_train_losses.npy',train_losses)
    np.save(prefix + '_test_accs.npy',test_accs)
    np.save(prefix + '_train_accs.npy',train_accs)
    torch.save(best_model.state_dict(),"saved_models/"+args.task_name+".pth")
