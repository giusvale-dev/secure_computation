from cifar_binary_dataset import CIFAR10CatBird, PoisonedDataset
from model import MobileNetBinaryClassifier
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

PRE_EVAL = True
TRAIN = True
EVAL = True
POISON = True

def predict(model, image_tensor, device):
    
    model.eval().to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)  # logits
        probs = torch.softmax(output, dim=1)
        confidence, pred_label = torch.max(probs, dim=1)

    return pred_label.item(), confidence.item()

def poison_image(model, target_img, base_img, learning_rate=1e-2, beta=1.0, max_iters=50, device='cuda'):

    '''
    Algorithm1 Poison Frogs
    '''
    model.eval().to(device)
    target_img = target_img.to(device)
    base_img = base_img.to(device)

    # target feature (f(t))
    with torch.no_grad():
        target_output = model(target_img).detach()

    # Init x = b
    x = base_img.clone().detach().requires_grad_(True)

    # Loop iterations
    for i in range(max_iters):
        
        # Forward step: compute gradient of Lp(x) = ||f(x) - f(t)||^2
        output = model(x)
        loss = F.mse_loss(output, target_output)
        grad = torch.autograd.grad(loss, x, create_graph=False)[0]

        # Gradient descent step
        x_bi = x - learning_rate * grad

        # Backward step: regularize toward base image
        x = (x_bi + learning_rate * beta * base_img) / (1 + beta * learning_rate)
        x = x.detach().clone().requires_grad_(True)

    return x.detach()

def save_model(model, filename="mobilenetv2_catbird.pth"):
    torch.save(model.state_dict(), filename)

def load_model(model, filename="mobilenetv2_catbird.pth"):
    model.load_state_dict(torch.load(filename))

def get_image_data_by_label(data_loader, label):
    
    for images, labels in data_loader:
        for i in range(len(labels)):
            if labels[i].item() == label:
                return images[i], labels[i].item()  
    return None, None

def denormalize(image_tensor, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    """
    Denormalizes the image tensor
    """
    if image_tensor.dim() == 3:  # For a single image (C, H, W)
        for i in range(3):  # For each color channel: R, G, B
            image_tensor[i, :, :] = image_tensor[i, :, :] * std[i] + mean[i]
    elif image_tensor.dim() == 4:  # For a batch of images (B, C, H, W)
        for i in range(3):  # For each color channel: R, G, B
            image_tensor[:, i, :, :] = image_tensor[:, i, :, :] * std[i] + mean[i]
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions")

def normalize(image_tensor, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    """
    Normalizes the image tensor
    """
    if image_tensor.dim() == 3:  # Single image (C, H, W)
        for i in range(3):  # For each color channel: R, G, B
            image_tensor[i, :, :] = (image_tensor[i, :, :] - mean[i]) / std[i]
    elif image_tensor.dim() == 4:  # Batch of images (B, C, H, W)
        for i in range(3):  # For each color channel: R, G, B
            image_tensor[:, i, :, :] = (image_tensor[:, i, :, :] - mean[i]) / std[i]
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions")
    
def plot_image(image_tensor, label, filename="default_image"):
    
    # Convert the tensor to a NumPy array and denormalize (if necessary)
    image = image_tensor.permute(1, 2, 0).cpu().numpy()  # Rearrange dimensions to HxWxC
    image = np.clip(image, 0, 1)  # Ensure the values are within [0, 1] for display

    # Plot the image
    plt.imshow(image)
    plt.title(f"Label: {'Bird' if label == 1 else 'Cat'}")
    plt.axis('off')
    plt.savefig(filename)

def plot_two_images(img1, img2, label1="Image 1", label2="Image 2", filename = "comparison"):
    
    # Remove batch dimension if present
    if img1.dim() == 4:
        img1 = img1.squeeze(0)
    if img2.dim() == 4:
        img2 = img2.squeeze(0)

    # Clamp values to [0, 1] just in case
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # Convert to HWC format for matplotlib
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img1_np)
    axes[0].set_title(label1)
    axes[0].axis("off")

    axes[1].imshow(img2_np)
    axes[1].set_title(label2)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(filename)

def evaluate(model, data_loader, device):
    model.eval().to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

def train(model, train_loader, device, epochs=5, lr=1e-3):

    criterion = torch.nn.CrossEntropyLoss() # Ensure that we do not need to insert a softmax layer (using this loss function)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        print(f"Process epoch {epoch + 1} of {epochs}...")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    return train_loss, train_acc, total

def main():

    print("Preparing Cat vs Bird CIFAR-10 Dataset...")
    dataset = CIFAR10CatBird()
    train_loader, test_loader = dataset.get_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing MobileNetV2 binary classifier...")
    model = MobileNetBinaryClassifier(freeze_features=True).to(device)

    if PRE_EVAL:
        print("Evaluating model before training...")
        acc = evaluate(model, test_loader, device)
        print(f"Pretraining (ImageNet) accuracy on Cat vs Bird test set: {acc * 100:.2f}%")
    
    if TRAIN:
        # For 1 epoch near 8 minutes
        start_time = time.time()
        print("Training model...")
        train_loss, train_acc, total = train(model, train_loader, device, epochs=1)
        print("Saving model")
        save_model(model)
        end_time = time.time()
        print(f"Training time: {(end_time - start_time)}s")
    
    if EVAL:
        load_model(model)
        print("Evaluation model...")
        test_acc = evaluate(model, test_loader, device)
        print(f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc*100:.2f}% "
              f"Test Acc: {test_acc*100:.2f}%")
    
    if POISON:
        # Step 5: Pick the normalized images
        base_image, base_label = get_image_data_by_label(test_loader, label=0)  # Cat
        target_image, target_label = get_image_data_by_label(test_loader, label=1)  # Bird

        denormalize(base_image)
        denormalize(target_image)

        # Step 6: Plot the picked images
        plot_two_images(base_image, target_image, "base_image", "target_image", "base_vs_target")

        # Normalize the images for poisoning
        normalize(base_image)
        normalize(target_image)

        # Poisoning image
        poison_img = poison_image(model, target_image.unsqueeze(0), base_image.unsqueeze(0), learning_rate=0.01, beta=0.7, max_iters=1000, device=device)

        # Denomarlize
        denormalize(poison_img)
        denormalize(base_image)

        plot_two_images(base_image, poison_img, "base_image", "poison", "base_vs_poison")

        # Normalize the poison image
        normalize(poison_img)

        label, conf = predict(model, poison_img, device)
        print(f"Predicted label: {'bird' if label == 1 else 'cat'} (confidence: {conf*100:.2f}%)")

        # Step 7: Add the poisoned image to the training dataset
        train_dataset = train_loader.dataset
        poisoned_dataset = PoisonedDataset(train_dataset, poison_img, poison_label=0)
        train_loader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=64, shuffle=True)

        # Step 8: new evaluation
        start_time = time.time()
        print("Training poisoned model...")

        train_loss, train_acc, total = train(model, train_loader, device, epochs=1)
        print("Saving model")
        save_model(model, filename="mobilenetv2_catbird_poisoned.pth")

        end_time = time.time()
        print(f"Training time: {(end_time - start_time)}s")
        
        load_model(model, filename="mobilenetv2_catbird_poisoned.pth")
        print("Evaluate model")
        test_acc = evaluate(model, test_loader, device)
        print(f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc*100:.2f}% "
              f"Test Acc: {test_acc*100:.2f}%")
        
if __name__ == "__main__":
    main()