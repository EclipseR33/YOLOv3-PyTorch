import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import tqdm
import os
import argparse

from DataSet.ILSVRC2012_dataset import ILSVRCDataSet
from Model.darknet import darknet_with_fc


def train():
    batch_size = opt.batch_size
    num_epochs = opt.num_epochs
    start_epoch = opt.start_epoch
    lr = opt.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    print('DataSet: ', end='')
    dataset_path = 'DataSet/ILSVRCDataSet.pt'
    if os.path.exists(dataset_path):
        dataset = torch.load(dataset_path)
        print('Load from {}'.format(dataset_path))
    else:
        dataset = ILSVRCDataSet(r'D:\AI\Dataset\ILSVRC2012', transform=transform)
        print('ILSVRCDataSet')
        torch.save(dataset, dataset_path)
    dataset.device = device
    dataset.transform = transform

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = darknet_with_fc(num_classes=10)
    model.to(device)

    if len(opt.weight_path) > 0:
        # model.load_state_dict(torch.load('Weights/DarkNet53_epoch_{}.pt'.format(start_epoch)))
        model.load_state_dict(torch.load(opt.weight_path))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0
        count = 0

        pbar = tqdm.tqdm(dl)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(out, dim=1)
            count += torch.sum(pred == y).to('cpu').item()
            running_loss += loss.item()

            pbar.set_description('Epoch: {} loss:{:.5f}'.format(epoch, loss.item()))

        # calculate epoch accuracy and loss
        train_acc = count * 100 / len(dataset)
        running_loss = running_loss / len(dl)
        print('Acc:{:.2f}%, Loss:{:.5f}'.format(train_acc, running_loss))

        # save model
        model_path = 'Weights/DarkNet53_epoch_{}.pt'.format(epoch)
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_path', type=str, default='')

    opt = parser.parse_args()

    train()
