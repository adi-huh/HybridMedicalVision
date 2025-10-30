"""
Training script for medical pretrained models

This script trains a model that's already pretrained on 200K+ chest X-rays,
so it requires less training time and achieves higher accuracy!
"""

import os
import torch
from medical_pretrained_model import (
    MedicalPretrainedModel,
    MedicalModelWithAttention,
    train_medical_model
)
from medical_dataset import get_medical_dataloaders
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


def evaluate_model(model, dataloader, device='cuda'):
    """Comprehensive evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, 
                                target_names=['Normal', 'Pneumonia'],
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_medical.png', dpi=150)
    print(f"\n✓ Confusion matrix saved to 'confusion_matrix_medical.png'")
    plt.close()
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"AUC Score: {auc:.4f}")
    print("="*60)
    
    return accuracy


def main():
    """Main training function"""
    
    # ===== CONFIGURATION =====
    config = {
        'data_dir': 'data',
        'batch_size': 16,  # Reduced for M2 Air
        'num_epochs': 20,  # Reduced from 30 for M2
        'num_workers': 0,  # MPS requires 0 workers
        'device': 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'),
        'model_type': 'basic',  # Use basic model (faster)
        'model_name': 'densenet121-res224-all',
        'freeze_backbone': True,  # Freeze backbone for faster training
        'grayscale': True,
    }
    
    print("\n" + "="*70)
    print("MEDICAL PRETRAINED MODEL TRAINING")
    print("="*70)
    print(f"Device: {config['device']}")
    print(f"Model: {config['model_name']}")
    print(f"Model Type: {config['model_type']}")
    print(f"Backbone Frozen: {config['freeze_backbone']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print("="*70)
    
    # ===== LOAD DATA =====
    print("\n📂 Loading datasets...")
    train_loader, val_loader, test_loader = get_medical_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        grayscale=config['grayscale']
    )
    
    # ===== CREATE MODEL =====
    print("\n🏗️  Creating model...")
    
    if config['model_type'] == 'with_attention':
        model = MedicalModelWithAttention(
            model_name=config['model_name'],
            num_classes=2
        )
    else:
        model = MedicalPretrainedModel(
            model_name=config['model_name'],
            num_classes=2,
            freeze_backbone=config['freeze_backbone']
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    # ===== TRAIN MODEL =====
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    
    import time
    start_time = time.time()
    
    model = train_medical_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        device=config['device']
    )
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Training completed in {training_time/60:.2f} minutes")
    
    # ===== TEST MODEL =====
    print("\n" + "="*70)
    print("🔬 EVALUATING ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load('models/medical_best_model.pth', map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config['device'])
    
    test_accuracy = evaluate_model(model, test_loader, device=config['device'])
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)
    print(f"Best Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"Model saved to: models/medical_best_model.pth")
    print("="*70)
    
    # ===== SAVE MODEL INFO =====
    model_info = {
        'config': config,
        'val_accuracy': checkpoint['val_acc'],
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    import json
    with open('models/medical_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    print("\n✓ Model info saved to: models/medical_model_info.json")
    
    return model


if __name__ == '__main__':
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run training
    model = main()
    
    print("\n" + "="*70)
    print("🎉 ALL DONE! Your medical model is ready to use!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run the app: python medical_app.py")
    print("2. Or use the model in your code:")
    print("   from medical_pretrained_model import load_medical_model")
    print("   model = load_medical_model('models/medical_best_model.pth')")
    print("="*70)