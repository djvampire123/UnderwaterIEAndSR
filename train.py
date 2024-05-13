import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from uwcc import uwcc
import os
import shutil
import sys
from unified_model import UnifiedEnhanceSuperResNet

def main():
    best_loss = 9999.0

    lr = 0.001
    batchsize = 1
    n_workers = 2
    epochs = 3000
    ori_fd = sys.argv[1]
    ucc_fd = sys.argv[2]
    ori_dirs = [os.path.join(ori_fd, f) for f in os.listdir(ori_fd)]
    ucc_dirs = [os.path.join(ucc_fd, f) for f in os.listdir(ucc_fd)]

    # Create model
    model = UnifiedEnhanceSuperResNet(upscale_factor=2)
    model = nn.DataParallel(model)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define criterion
    criterion = nn.MSELoss()

    # Load data
    trainset = uwcc(ori_dirs, ucc_dirs, train=True)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=n_workers)

    # Train
    for epoch in range(epochs):
        tloss = train(trainloader, model, optimizer, criterion, epoch)
        print(f'Epoch:[{epoch}/{epochs}] Loss {tloss}')
        is_best = tloss < best_loss
        best_loss = min(tloss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print('Best Loss:', best_loss)

def train(trainloader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    model.train()

    for i, sample in enumerate(trainloader):
        ori, ucc = sample
        ori = ori.cuda()
        ucc = ucc.cuda()

        corrected = model(ori)
        loss = criterion(corrected, ucc)
        losses.update(loss.item(), ori.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg

def save_checkpoint(state, is_best):
    """Saves checkpoint to disk"""
    freq = 500
    epoch = state['epoch']
    filename = './checkpoints/model_tmp.pth.tar'
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    torch.save(state, filename)
    if epoch % freq == 0:
        shutil.copyfile(filename, f'./checkpoints/model_{epoch}.pth.tar')
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
