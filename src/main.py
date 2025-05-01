import torch
import matplotlib.pyplot as plt
import numpy as np
from neural_network import Net
import torch.optim as optim
import torch.nn as nn
from dataloader import BinaryCIFAR10Dataset
from poison import generate_poison, calculate_beta
from torchvision.transforms.functional import to_pil_image
from constants import DEVICE
from constants import NUM_POISONS

def pick_target_image(trainloader: BinaryCIFAR10Dataset):
    """
    Picks the first image with label == 1 from a test DataLoader.

    Args:
        trainloader (dataloader.DataLoader): 
            The dataloader containing Bird vs Cat images

    Returns:
        torch.Tensor:
            A single image tensor of shape [1, channels, height, width] corresponding to the first occurrence
            of label == 1 in the dataset.
            Returns None if no image with label == 1 is found.
    """
    target_img = None
    for imgs, lbls in trainloader:
        for i in range(len(lbls)):
            if lbls[i].item() == 1:
                target_img = imgs[i].unsqueeze(0).to(DEVICE)
                break
        if target_img is not None:
            break
    return target_img

def pick_multiple_base_images(trainloader: BinaryCIFAR10Dataset, counter: int):
    """
    Selects multiple images with label == 0 from the training DataLoader.

    Args:
        trainloader (dataloader.DataLoader): 
            The dataloader containing Bird vs Cat images
        counter (int):
            The number of base images to collect.

    Returns:
        tuple:
            - base_imgs (list of torch.Tensor): 
                A list containing `counter` image tensors, each with shape [1, channels, height, width], 
                corresponding to images where label == 0. Each tensor is moved to the DEVICE.
            - base_indices (list of int): 
                A list containing the flattened dataset indices of the selected base images.

    Notes:
        - The function stops searching once the required number (`counter`) of base images is collected.
        - Assumes that each batch returned by `trainloader` is a tuple (images, labels),
          where images have shape [batch_size, channels, height, width] and labels have shape [batch_size].
        - Flattened index is computed as: `batch_index * batch_size + sample_index_within_batch`.
    """
    base_imgs = []
    base_indices = []
    for idx, (imgs, lbls) in enumerate(trainloader):
        for i in range(len(lbls)):
            if lbls[i].item() == 0:
                base_imgs.append(imgs[i].unsqueeze(0).to(DEVICE))
                flat_idx = idx * trainloader.batch_size + i
                base_indices.append(flat_idx)
                if len(base_imgs) >= counter:
                    break
        if len(base_imgs) >= counter:
            break
    return base_imgs, base_indices
    
def main():
    
    # Load data
    dataloader = BinaryCIFAR10Dataset(batch_size=64)
    trainloader = dataloader.trainloader
    testloader = dataloader.testloader

    # Initialize model
    net = Net().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-4)

    # Pick the target image
    target_img = pick_target_image(trainloader)

    # Pick NUM_POISONS base images
    base_imgs, base_indices = pick_multiple_base_images(trainloader, NUM_POISONS)

    base_pil_list = []
    poison_pil_list = []
    used_indices = []

    misclassified_images = []
    misclassified_indices = []
    misclassified_preds = []
    misclassified_labels = []

    print("[*] Training clean model...")
    net.train_network(trainloader, optimizer, criterion, num_epochs=20)
    
    # Test on clean set
    print("[*] Evaluating on clean test set...")
    net.test(testloader)

    print("Start the poison attack")
    print(f"[*] Generating and injecting {NUM_POISONS} poisoned samples...")
    
    for i, (base_img, index) in enumerate(zip(base_imgs, base_indices)):

        beta = calculate_beta(beta0=0.25)
        poisoned_img = generate_poison(model=net, target_instance=target_img, base_instance=base_img, learning_rate=0.01, max_iters=1500, beta=beta, device=DEVICE)

        poisoned_img = poisoned_img.clamp(0, 1).detach().cpu().squeeze()
        pil_poison = to_pil_image(poisoned_img)
        pil_base = to_pil_image(base_img.cpu().squeeze())

        # Replace image and label in dataset
        dataloader.trainset.data[index] = np.array(pil_poison)
        dataloader.trainset.targets[index] = 0  # Poisoned label

        # Save for plot
        base_pil_list.append(pil_base)
        poison_pil_list.append(pil_poison)
        used_indices.append(index)

        # Check misclassification
        with torch.no_grad():
            output = net(poisoned_img.unsqueeze(0).to(DEVICE))
            prediction = torch.sigmoid(output).cpu().item()
            predicted_class = int(prediction >= 0.5)

        if predicted_class != 0:
            misclassified_images.append(pil_poison)
            misclassified_indices.append(index)
            misclassified_preds.append(predicted_class)
            misclassified_labels.append(0)  # True class is 0

        percent = (i + 1) / NUM_POISONS * 100
        print(f"\rPoisoning progress: {percent:.2f}%", end='', flush=True)

    print("")

    # Show misclassified poisoned images
    print(f"[*] {len(misclassified_images)} poisoned images misclassified.")
    if len(misclassified_images) > 0:
        
        fig, axs = plt.subplots(len(misclassified_images), 1, figsize=(6, len(misclassified_images) * 2))
        if len(misclassified_images) == 1:
            axs = [axs]  # wrap in list for consistent indexing
        for i in range(len(misclassified_images)):
            axs[i].imshow(misclassified_images[i])
            original_label = "Bird" if misclassified_labels[i] == 1 else "Cat"
            predicted_label = "Bird" if misclassified_preds[i] == 1 else "Cat"
            axs[i].set_title(f"Misclassified Poisoned Image\n"                  
            f"Original Label: {original_label} -> Predicted: {predicted_label}")
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig("missclassified")

    # ------------------------------------------------------------
    # Plot poisoned images vs. base images for visual check
    # ------------------------------------------------------------
    print("[*] Visualizing poisoned vs. base images...")

    fig, axs = plt.subplots(NUM_POISONS, 2, figsize=(6, NUM_POISONS * 3))

    for i in range(NUM_POISONS):
        axs[i, 0].imshow(base_pil_list[i])
        axs[i, 0].set_title(f"Base Image #{i}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(poison_pil_list[i])
        axs[i, 1].set_title(f"Poisoned Image #{i}")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("base_vs_poisoned.png")

    # Init the poisoned model
    poisoned_net = Net().to(DEVICE)
    poisoned_optimizer = optim.SGD(poisoned_net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    poisoned_criterion = nn.BCEWithLogitsLoss()

    print("[*] Training on poisoned dataset...")
    poisoned_net.train_network(trainloader, poisoned_optimizer, poisoned_criterion, num_epochs=20)

    print("[*] Evaluating poisoned model...")
    poisoned_net.test(testloader)

if __name__=="__main__":
    main()