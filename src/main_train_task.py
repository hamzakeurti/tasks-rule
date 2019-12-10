import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import LOG_INFO
from model import Encoder,Classification_Decoder

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
LOG_INTERVAL = 50
EPOCHS = 20

LR = 0.01
MM = 0.8
WD = 0.0
Encoder_out = 512
Decoder_out = 51


device = 'cuda:0'
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader
test_loader

encoder = Encoder()
decoder = Classification_Decoder(Encoder_out,Decoder_out).to(device)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.SGD(params, lr=LR, momentum=MM, weight_decay=WD)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_list = []
    acc_list = []
    saved_loss_list = []
    saved_acc_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output.view(-1,512)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).float().mean()
        acc_list.append(acc.item())
        if batch_idx % LOG_INTERVAL == 0:
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(loss_list), np.mean(acc_list))
            LOG_INFO(msg)
            saved_loss_list.append(np.mean(loss_list))
            saved_acc_list.append(np.mean(acc_list))
            loss_list.clear()
            acc_list.clear()
    return saved_loss_list,saved_acc_list


def test(model, device, test_data,test_label):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data, target = test_data.to(device), test_label.to(device)
        data = data.view(data.shape[0],1,28,28)
        output = model(data)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_data),
        100. * correct / len(test_data)))
    return test_loss,correct/len(test_data)

if __name__=='__main__':
    train_losses,train_accs,test_losses,test_accs = [],[],[],[]
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss,test_acc = test(model, device, test_data,test_label)
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    prefix = 'CNN_'
    np.save(prefix + 'test_losses.npy',test_losses)
    np.save(prefix + 'train_losses.npy',train_losses)
    np.save(prefix + 'test_accs.npy',test_accs)
    np.save(prefix + 'train_accs.npy',train_accs)