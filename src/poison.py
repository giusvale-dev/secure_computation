import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


# def generate_poison(model, target_img, base_img, max_iters=1500, lr=0.01, obj_threshold=0.5, device='cuda'):
#     """
#     Algorithm 1 from the Poison Frogs paper.

#     Inputs:
#         model           frozen feature extractor (e.g. Inception up to penultimate layer)
#         target_img      (1,C,H,W) tensor of the true target instance
#         base_img        (1,C,H,W) tensor of the base instance to poison
#         max_iters       maximum GD steps
#         lr              step size λ in Alg. 1
#         obj_threshold   early stop if feature loss < threshold
#         device          'cuda' or 'cpu'
#     Returns:
#         poison_img       the crafted poison (1,C,H,W), clamped to [0,1]
#         final_feat_diff  final feature space MSE to target
#     """
#     model.eval()
#     target_img = target_img.to(device)
#     base_img   = base_img.to(device)

#     # x0 ← b
#     poison_img = base_img.clone().detach().to(device).requires_grad_(True)

#     # Precompute f(t)
#     with torch.no_grad():
#         target_feat = model(target_img).detach()

#     for i in range(max_iters):
#         # forward: compute feature MSE loss
#         poison_feat = model(poison_img)
#         feat_loss   = F.mse_loss(poison_feat, target_feat)

#         # gradient step: xi = xi−1 − λ ∇x Lp
#         grad = torch.autograd.grad(feat_loss, poison_img)[0]
#         with torch.no_grad():
#             poison_img -= lr * grad
            
#             # perturbation = poison_img - base_img
#             # epsilon = 0.05  # small value like 5/255
#             # perturbation = torch.clamp(perturbation, -epsilon, epsilon)
#             # poison_img = base_img + perturbation
            
#             poison_img.clamp_(0, 1)   # keep valid image range

#         # re attach grad for next iteration
#         poison_img.requires_grad_(True)

#         if feat_loss.item() < obj_threshold:
#             print(f"[Early stop] iter={i}, feat_loss={feat_loss.item():.4f}")
#             break

#     return poison_img.detach(), feat_loss.item()

# def generate_poison(model, target_input, base_input, 
#                                learning_rate=0.01, max_iters=100, 
#                                projection_fn=None, device='cpu'):
#     """
#     Generate a poisoned example using gradient-based optimization.
    
#     Args:
#         model: A PyTorch model.
#         target_input: The target input tensor (1 x C x H x W).
#         base_input: The base input tensor to start from (1 x C x H x W).
#         learning_rate: Step size for gradient descent.
#         max_iters: Number of optimization iterations.
#         projection_fn: Optional function to project x back to valid input space.
#         device: 'cpu' or 'cuda'.
        
#     Returns:
#         A poisoned example tensor.
#     """
#     model.eval()
#     x = base_input.clone().detach().to(device).requires_grad_(True)
#     t = target_input.clone().detach().to(device)
    
#     for i in range(max_iters):
#         model.zero_grad()
        
#         # Forward pass
#         f_x = model(x)
#         f_t = model(t)

#         # Loss: squared L2 distance between f(x) and f(t)
#         loss = F.mse_loss(f_x, f_t)

#         # Backward step
#         loss.backward()

#         # Gradient descent step
#         with torch.no_grad():
#             x -= learning_rate * x.grad
#             x.grad.zero_()

#             # Optional: project x to valid input space (e.g., clip to [0, 1])
#             if projection_fn:
#                 x[:] = projection_fn(x)

#     return x.detach()

def generate_poison(net, target_img, base_img, max_iters=1500, lr=0.01, beta0=0.25, obj_threshold=1e-3, device='cuda'):
    """
    Generate a poisoned image using the provided algorithm.
    
    Parameters:
    - net: The neural network model.
    - target_img: The target instance to misclassify.
    - base_img: The base instance (original image).
    - max_iters: Maximum number of iterations to run the algorithm.
    - lr: Learning rate for the update step.
    - beta0: Parameter β in the backward step update.
    - obj_threshold: Threshold for the objective function for early stopping.
    - device: Device to run the calculations ('cuda' or 'cpu').
    
    Returns:
    - poisoned_img: The generated poisoned image.
    """
    
    # Ensure both images are on the same device
    target_img = target_img.to(device)
    base_img = base_img.to(device)

    # Initialize poisoned image as the base image
    poisoned_img = base_img.clone().detach().requires_grad_(True)
    
    # Define the loss function: Lp(x) = ||f(x) - f(t)||^2
    criterion = nn.MSELoss()

    # Set up optimizer for the poisoned image
    optimizer = optim.SGD([poisoned_img], lr=lr)

    # Initialize beta
    beta = beta0
    
    for i in range(max_iters):
        # Forward pass: Get the model's prediction for poisoned image
        optimizer.zero_grad()
        
        # Get the output from the network for the poisoned image
        output = net(poisoned_img)
        
        # Calculate the objective function Lp(x) = ||f(x) - f(t)||^2
        loss = criterion(output, net(target_img))
        
        # Check if the loss is below the threshold for early stopping
        if loss.item() < obj_threshold:
            print(f"Early stopping at iteration {i+1} with loss {loss.item()}")
            break

        # Backward pass: Compute gradients
        loss.backward()

        # Check if gradients are available
        if poisoned_img.grad is None:
            print(f"Warning: Gradients are None at iteration {i+1}")
            break

        # Forward step (xi = xi-1 - λ∇x Lp(xi-1))
        with torch.no_grad():
            poisoned_img -= lr * poisoned_img.grad
        
        # Backward step (xi = (xbi + λβb) / (1 + βλ))
        with torch.no_grad():
            poisoned_img = (poisoned_img + lr * beta * base_img) / (1 + beta * lr)
        
        # Optional: Print the loss for each iteration (for monitoring)
        if i % 100 == 0:
            print(f"Iteration {i+1}/{max_iters}, Loss: {loss.item()}")

    return poisoned_img

def clip_projection(x):
    return x.clamp(0, 1)