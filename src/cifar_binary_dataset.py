from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch

class FilteredCIFAR10Binary(Dataset):
    def __init__(self, base_dataset, cat_label=3, bird_label=2):
        self.base_dataset = base_dataset
        self.cat_label = cat_label
        self.bird_label = bird_label

        # Precompute filtered indices
        self.filtered_indices = [
            i for i in range(len(base_dataset))
            if base_dataset.targets[i] in [self.cat_label, self.bird_label]
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        real_idx = self.filtered_indices[idx]
        image, label = self.base_dataset[real_idx]

        # Relabel: cat → 0, bird → 1
        if label == self.cat_label:
            return image, torch.tensor(0)
        else:
            return image, torch.tensor(1)

class CIFAR10CatBird:
    def __init__(self, root='./data', batch_size=32):
        self.root = root
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._prepare_loaders()

    def _prepare_loaders(self):
        full_train = datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=self.transform)
        full_test = datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=self.transform)

        train_binary = FilteredCIFAR10Binary(full_train)
        test_binary = FilteredCIFAR10Binary(full_test)

        self.train_loader = DataLoader(train_binary, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_binary, batch_size=self.batch_size)

    def get_loaders(self):
        return self.train_loader, self.test_loader

class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, poison_image, poison_label):
        self.base_dataset = base_dataset
        self.poison_image = poison_image.squeeze(0)  # shape: (3, 32, 32)
        self.poison_label = poison_label

    def __len__(self):
        return len(self.base_dataset) + 1

    def __getitem__(self, idx):
        if idx == len(self.base_dataset):
            return self.poison_image, torch.tensor(self.poison_label)
        else:
            img, label = self.base_dataset[idx]
            return img, label 