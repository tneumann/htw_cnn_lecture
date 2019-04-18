import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

from poutyne.framework import Model
from poutyne.framework.callbacks import TensorBoardLogger

from tensorboardX import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(7*7*32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        # TODO: instantiate Conv2d and Linear layers
#        # self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
#
#    def forward(self, x):
#        # pass x through layers
#        # x = F.relu(self.conv1(x))
#        # use F.max_pool2d for pooling
#        # TODO
#        return x


def main():
    lr = 1.e-1
    batch_size = 128
    num_epochs = 10

    # load data
    train_data = datasets.MNIST(
        root='./data', train=True, 
        download=True, transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=batch_size)

    valid_data = datasets.MNIST(
        root='./data', train=False, 
        download=True, transform=transforms.ToTensor()
    )
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
    torch.save(net.state_dict(), 'mnist_cnn.pth')



if __name__ == '__main__':
    main()
