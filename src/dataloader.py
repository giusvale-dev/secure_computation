import torch
import torchvision
import torchvision.transforms as transforms

class Dataloader:
    
    def __init__(self, target_class=3, batch_size=64):
        self.target_class = target_class
        self.transform = transforms.ToTensor()
        
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform)

        # Convert to binary targets (1 = target_class, 0 = others)
        self.trainset.targets = [self.binary_target(t) for t in self.trainset.targets]
        self.testset.targets = [self.binary_target(t) for t in self.testset.targets]

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size, shuffle=False, num_workers=2)

    def binary_target(self, target):
        return 1 if target == self.target_class else 0

