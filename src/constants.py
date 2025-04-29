import torch

TRAINED_MODEL_PATH = 'data/cifar_net.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_POISONS=5