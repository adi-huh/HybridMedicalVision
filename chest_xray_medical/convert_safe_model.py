"""
Script to convert existing model to safe format
"""

import torch
from medical_pretrained_model import MedicalPretrainedModel
from safe_model_utils import save_model_safely

def convert_model():
    try:
        # Load the existing model
        device = 'cpu'  # Convert on CPU for safety
        print("Loading existing model...")
        checkpoint = torch.load('Models/medical_best_model.pth', map_location=device)
        
        # Create a new model instance
        print("Creating new model instance...")
        model = MedicalPretrainedModel(
            model_name='densenet121-res224-all',
            num_classes=2,
            freeze_backbone=True
        )
        
        # Load state dict
        print("Loading state dictionary...")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save in safe format
        print("Saving in safe format...")
        save_model_safely(
            model=model,
            optimizer=None,  # We don't need optimizer for inference
            val_acc=checkpoint['val_acc'] if 'val_acc' in checkpoint else 0.0,
            epoch=checkpoint.get('epoch', 0),
            filename='Models/medical_best_model_safe.pth'
        )
        
        print("✅ Model converted and saved safely!")
        print("New model file: Models/medical_best_model_safe.pth")
        print("Model info saved: Models/medical_best_model_safe_info.json")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise e
    
    print("✅ Model converted and saved safely!")
    print("New model file: Models/medical_best_model_safe.pth")
    print("Model info saved: Models/medical_best_model_safe_info.json")

if __name__ == '__main__':
    convert_model()