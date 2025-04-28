import torch
import torch.nn.functional as F
from torchvision import transforms
import copy

def generate_poison(model, target_img, base_img,
                    max_iters=1500, lr=0.01,
                    coeff_sim_inp=0.2,  # kept for signature compatibility, not used
                    obj_threshold=1e-3, device='cuda'):
    """
    Algorithm 1 (Poisoning Example Generation) from the Poison Frogs paper.

    Inputs:
        model           frozen feature extractor (e.g. Inception up to penultimate layer)
        target_img      (1,C,H,W) tensor of the true target instance
        base_img        (1,C,H,W) tensor of the base instance to poison
        max_iters       maximum GD steps
        lr              step size λ in Alg. 1
        obj_threshold   early stop if feature loss < threshold
        device          'cuda' or 'cpu'
    Returns:
        poison_img       the crafted poison (1,C,H,W), clamped to [0,1]
        final_feat_diff  final feature space MSE to target
    """
    model.eval()
    target_img = target_img.to(device)
    base_img   = base_img.to(device)

    # x0 ← b
    poison_img = base_img.clone().detach().to(device).requires_grad_(True)

    # Precompute f(t)
    with torch.no_grad():
        target_feat = model(target_img).detach()

    for i in range(max_iters):
        # forward: compute feature MSE loss
        poison_feat = model(poison_img)
        feat_loss   = F.mse_loss(poison_feat, target_feat)

        # gradient step: xi = xi−1 − λ ∇x Lp
        grad = torch.autograd.grad(feat_loss, poison_img)[0]
        with torch.no_grad():
            poison_img -= lr * grad
            poison_img.clamp_(0, 1)   # keep valid image range

        # re attach grad for next iteration
        poison_img.requires_grad_(True)

        if feat_loss.item() < obj_threshold:
            print(f"[Early stop] iter={i}, feat_loss={feat_loss.item():.4f}")
            break

    return poison_img.detach(), feat_loss.item()



# def generate_poison(model, target_img, base_img, max_iters=1500, lr=0.01, coeff_sim_inp=0.2, obj_threshold=2.9, device='cuda'):
#     """
#     One-shot poison generation (from Poison Frogs paper).
#     Matches target in feature space while remaining visually close to base image.
    
#     Inputs:
#         - model: the frozen feature extractor (e.g., Inception up to penultimate layer)
#         - target_img: (1, C, H, W) image tensor
#         - base_img: (1, C, H, W) image tensor (to be poisoned)
#     Returns:
#         - poison image tensor (1, C, H, W)
#         - final feature difference (float)
#     """
#     model.eval()
#     target_img = target_img.to(device)
#     base_img = base_img.to(device)
    
#     poison_img = base_img.clone().detach().to(device).requires_grad_(True)

#     # Get target features once
#     with torch.no_grad():
#         target_feat = model(target_img)

#     for i in range(max_iters):
#         poison_feat = model(poison_img)

#         feat_diff = F.mse_loss(poison_feat, target_feat)
#         img_diff = F.mse_loss(poison_img, base_img)

#         loss = feat_diff + coeff_sim_inp * img_diff

#         grad = torch.autograd.grad(loss, poison_img)[0]

#         with torch.no_grad():
#             poison_img -= lr * grad
#             poison_img.clamp_(0, 1)  # clamp to valid range

#         poison_img.requires_grad_(True)

#         # if i % 100 == 0 or i == max_iters - 1:
#         #     print(f"[{i}] Feature loss: {feat_diff.item():.4f} | Img sim loss: {img_diff.item():.4f} | Total: {loss.item():.4f}")

#         # Optional early stopping
#         if feat_diff.item() < obj_threshold:
#             print(f"Early stop at iter {i} with feature diff {feat_diff.item():.4f}")
#             break

#     return poison_img.detach(), feat_diff.item()


# def generate_poison(model, target_img, base_img, max_iters=1000, lr=0.01, device='cuda'):
#     """
#     Poisoning example generation as in Algorithm 1 from Poison Frogs.
#     Inputs:
#         - model: the feature extractor or classifier (with frozen weights)
#         - target_img: target instance image tensor (1, C, H, W)
#         - base_img: base image to poison (1, C, H, W)
#     """
#     model.eval()
#     target_img = target_img.to(device)
#     base_img = base_img.clone().detach().to(device).requires_grad_(True)

#     # Extract target feature
#     with torch.no_grad():
#         target_feature = model(target_img)

#     for i in range(max_iters):
#         output = model(base_img)
#         loss = F.mse_loss(output, target_feature)

#         # Backward step
#         grad = torch.autograd.grad(loss, base_img)[0]
#         base_img = base_img - lr * grad
#         base_img = base_img.detach().clone().requires_grad_(True)

#     return base_img.detach()
