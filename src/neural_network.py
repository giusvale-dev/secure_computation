import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from constants import TRAINED_MODEL_PATH, classes
from sklearn.metrics import confusion_matrix, classification_report


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        


    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))     # ReLU after fc1
        x = self.relu(self.fc2(x))     # ReLU after fc2
        x = self.fc3(x)
       # x = self.sigmoid(x)            # Sigmoid at output
        return x

    def train_network(self, trainloader, optimizer, criterion):
        self.train()
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        torch.save(self.state_dict(), TRAINED_MODEL_PATH)
    
    def test(self, testloader):
        self.eval()  # set model to evaluation mode

        self.load_state_dict(torch.load(TRAINED_MODEL_PATH))  # no need for weights_only=True unless you're using torch.compile models

        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                labels = labels.float().unsqueeze(1)  # Ensure shape is [batch, 1] to match output

                outputs = self(images)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    
    def statistics(self, testloader):
        self.eval()  # set model to evaluation mode
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                labels = labels.float().unsqueeze(1)  # Ensure labels shape matches output
                outputs = self(images)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        print(f"Binary Classification Accuracy: {accuracy:.2f}%")

        # Optional: print confusion matrix
        
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["Non-cat", "Cat"]))

    