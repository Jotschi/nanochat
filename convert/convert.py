import torch
from safetensors.torch import save_file

# Load PyTorch .pt / .pth weights
state_dict = torch.load("model_006861.pt", map_location="cpu")

# If it's a full model, extract the state_dict
if hasattr(state_dict, "state_dict"):
    state_dict = state_dict.state_dict()

# Save as safetensors
save_file(state_dict, "model.safetensors")