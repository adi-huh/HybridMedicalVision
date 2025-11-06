"""
Utilities for safely saving and loading models
"""

import torch
import json
import os

def save_model_safely(model, optimizer, val_acc, epoch, filename):
    """
    Save model in a safe format that won't trigger security warnings
    """
    # Save only the state dict, not the entire model
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'val_acc': val_acc,
        'epoch': epoch
    }
    
    # Use new zipfile serialization
    torch.save(save_dict, filename, _use_new_zipfile_serialization=True)
    
    # Save model architecture info separately
    model_info = {
        'model_type': model.__class__.__name__,
        'num_classes': model.num_classes if hasattr(model, 'num_classes') else 2,
        'input_size': 224,  # Standard input size
        'channels': 1  # Medical X-rays are grayscale
    }
    
    info_file = os.path.splitext(filename)[0] + '_info.json'
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=4)

def load_model_safely(model_class, filename, device='cpu'):
    """
    Load model safely with architecture verification
    """
    # Load model info
    info_file = os.path.splitext(filename)[0] + '_info.json'
    with open(info_file, 'r') as f:
        model_info = json.load(f)
    
    # Create model instance
    model = model_class(
        num_classes=model_info['num_classes'],
        grayscale=model_info['channels'] == 1
    )
    
    # Load state dict
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint