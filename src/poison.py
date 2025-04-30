import torch

def calculate_beta(dimb = 3 * 32 * 32, beta0=0.25, feature_space_dim=3072):
    """
    Compute β for the poisoning attack.
    
    Args:
        dimb (int): The dimensionality of the base instance (image) (C * H * W). 3x32x32 CIFAR-10
        beta0 (float): The constant β0 used in the calculation. Default is 0.25.
        feature_space_dim (int): The dimensionality of InceptionV3's feature space representation layer. Default is 3072.

    Returns:
        float: The computed β.
    Notes:
        See the Poison Frogs paper for other details
    """
    # Apply the formula for β
    beta = beta0 * (feature_space_dim**2) / (dimb**2)
    return beta

def generate_poison(model, target_instance, base_instance, learning_rate=0.1, max_iters=100, beta=0.1, device="cuda"):
    """
    Poison attack Algorithm1 of Poison Frogs paper
    
    Args:
        model (nn.Module): The trained model.
        target_instance (Tensor): The target instance for poisoning.
        base_instance (Tensor): The base instance for poisoning.
        learning_rate (float): Learning rate for the optimization.
        max_iters (int): Number of iterations for the optimization.
        beta (float): The scaling factor for the backward step.
        device (str): Device to run the computation on ('cpu' or 'cuda').

    Returns:
        The poisoned image.
    """
    
    model = model.to(device)
    target_instance = target_instance.to(device)
    base_instance = base_instance.to(device)
    
    # Initialize x with the base instance and enable gradient tracking
    x = base_instance.clone().detach().requires_grad_(True).to(device)

    # Loss function (Lp(x) = || f(x) - f(t) ||^2)
    def Lp(x):
        output_x = model(x)
        output_target = model(target_instance)
        loss = torch.norm(output_x - output_target, p=2) ** 2
        return loss

    for i in range(max_iters):

        # x requires gradients every iteration
        x.requires_grad_()  
        
        loss = Lp(x)

        # gradients computation
        gradients = torch.autograd.grad(loss, x)[0]

        # Forward step: update x using gradient descent
        with torch.no_grad():
            xbi = x - learning_rate * gradients  # Perform update on x

            # Backward step: update x with the base instance influence
            x = (xbi + learning_rate * beta * base_instance) / (1 + beta * learning_rate)

            # Ensure x stays within the valid image range ([0, 1])
            x = torch.clamp(x, 0, 1)

    # Return the final poisoned image, detached from the computation graph
    return x.detach()