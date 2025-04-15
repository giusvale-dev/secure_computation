import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from neural_network import Net
import torch.optim as optim
import torch.nn as nn
from dataloader import Dataloader

TRAINING = True
PATH = 'data/cifar_net.pth'

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("data/pictures")

def main():

    dataloader = Dataloader(target_class=3, batch_size=64)

    trainloader = dataloader.trainloader
    testloader = dataloader.testloader

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # show images
    imshow(torchvision.utils.make_grid(images))
    
    net = Net()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if TRAINING:
        net.train_network(trainloader, optimizer, criterion)
    
    net.test(testloader)
    net.statistics(testloader)


if __name__=="__main__":
    main()