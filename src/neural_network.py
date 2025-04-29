import torch
import torch.nn as nn
from constants import TRAINED_MODEL_PATH

class Net(nn.Module):
    """
    A fully connected neural network for binary image classification.

    Architecture:
    - Input: 3x32x32 (CIFAR-10 image size)
    - Flatten layer
    - FC1: 3072 → 512 + ReLU
    - FC2: 512 → 256 + ReLU
    - FC3: 256 → 1 (logit output)
    """

    def __init__(self):
        
        """
        Initializes the network layers.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Tensor: Output logits of shape (batch_size, 1)
        """
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_network(self, trainloader, optimizer, criterion, num_epochs=100):
        
        """
        Trains the neural network.

        Args:
            trainloader (DataLoader): Dataloader for training data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (Loss): Loss function.
            num_epochs (int, optional): Number of training epochs. Default is 100.

        Returns:
            list: Average training loss per epoch.
        """

        self.train()
        train_losses = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            self.train()

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = labels.float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Record average training loss
            avg_train_loss = running_loss / len(trainloader)
            train_losses.append(avg_train_loss)

            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        torch.save(self.state_dict(), TRAINED_MODEL_PATH)
        return train_losses
    
    def test(self, testloader):
        """
        Evaluates the trained network on the test data.

        Args:
            testloader (DataLoader): Dataloader for test data.
        """
        self.eval()
        self.load_state_dict(torch.load(TRAINED_MODEL_PATH))
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                labels = labels.float().unsqueeze(1)  

                outputs = self(images)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy:.4f}%')