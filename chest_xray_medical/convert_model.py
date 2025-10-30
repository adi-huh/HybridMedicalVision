import torch
from safetensors.torch import save_file

checkpoint = torch.load('models/medical_best_model.pth')
save_file(checkpoint, 'models/medical_best_model.safetensors')
print("Model converted successfully!")
