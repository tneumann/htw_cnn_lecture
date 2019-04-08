# import numpy - array processing library
import numpy as np 

# pytorch imports:
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# pytorch functionality for image datasets
from torchvision import datasets, transforms

from poutyne.framework import Model
from poutyne.framework.callbacks import TensorBoardLogger

from tensorboardX import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 10, bias=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


def main():
    lr = 1.e-1
    batch_size = 128
    num_epochs = 5

    # load data
    train_data = datasets.MNIST(root='./data', train=True, 
            download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=batch_size)

    valid_data = datasets.MNIST(root='./data', train=False, 
            download=True, transform=transforms.ToTensor())
    valid_loader = torch.utils.data.DataLoader(
            valid_data, shuffle=False, batch_size=batch_size)

    # setup network
    net = Net()
    sgd = optim.SGD(net.parameters(), lr=lr)
    model = Model(net, sgd, 'cross_entropy', metrics=['accuracy']) 
    model.cuda() 

    # train!
    model.fit_generator(
            train_loader, valid_loader,
            epochs=num_epochs, 
            callbacks=[TensorBoardLogger(SummaryWriter())]
    )
    
    # save
    torch.save(net.state_dict(), 'mynet.pkl')



if __name__ == '__main__':
    main()
