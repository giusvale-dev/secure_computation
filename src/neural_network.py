import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from constants import TRAINED_MODEL_PATH


class Net(nn.Module):
    def __init__(self):
        #torch.use_deterministic_algorithms(True)
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
      #  self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
      #  self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))     # ReLU after fc1
       # x = self.dropout1(x)
        x = self.relu(self.fc2(x))     # ReLU after fc2
       # x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def train_network(self, trainloader, optimizer, criterion, num_epochs=100):

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
        self.eval()
        self.load_state_dict(torch.load(TRAINED_MODEL_PATH))  # no need for weights_only=True unless you're using torch.compile models
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
    
    def evaluate_loss(self, dataloader, criterion):
        self.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                labels = labels.float().unsqueeze(1)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(dataloader)