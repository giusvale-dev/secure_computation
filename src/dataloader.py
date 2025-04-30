import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class CustomCIFAR10Dataset(Dataset):
    """
    Custom dataset for CIFAR-10 binary classification (bird vs. cat).
    """
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        # Filter out images that are not of birds (class 2) or cats (class 3)
        self.data = []
        self.targets = []
        for img, lbl in self.dataset:
            if lbl == 2 or lbl == 3:  # Bird (2) or Cat (3)
                self.data.append(img)
                self.targets.append(lbl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, lbl = self.data[index], self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, lbl

class Dataloader:
    """
    A data loader class for the CIFAR-10 dataset that performs binary classification
    of bird vs. cat images.
    """

    def __init__(self, batch_size=64):
        """
        Initializes the Dataloader object.
        
        Args:
            batch_size (int): Number of samples per batch.
        """
        
        self.bird_class = 2
        self.cat_class = 3

        # Apply transformations
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Load CIFAR-10 dataset
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True)
        
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True)

        # Create custom datasets with filtering and transformations
        self.trainset = CustomCIFAR10Dataset(self.trainset, transform=self.transform)
        self.testset = CustomCIFAR10Dataset(self.testset, transform=self.transform)

         # Convert to binary targets (1 = target_class, 0 = others)
        self.trainset.targets = [self.binary_target(t) for t in self.trainset.targets]
        self.testset.targets = [self.binary_target(t) for t in self.testset.targets]

        # Create DataLoaders
        self.trainloader = DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        self.testloader = DataLoader(
            self.testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    def binary_target(self, target):
        """
        Convert CIFAR-10 class index to binary label.

        Args:
            target (int): Original CIFAR-10 class index.

        Returns:
            int: 1 if class is bird, 0 otherwise.
        """
        return 1 if target == self.bird_class else 0