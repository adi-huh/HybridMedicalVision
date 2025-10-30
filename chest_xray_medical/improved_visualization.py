import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_and_preprocess_image(image_path, img_size=224):
    """
    Load and preprocess image - convert to grayscale for model compatibility
    """
    # Read image with OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Add channel dimension for albumentations
    image_gray = np.expand_dims(image_gray, axis=-1)
    
    # Define preprocessing pipeline
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(clip_limit=2.0, p=1.0),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    # Keep original for visualization (convert to PIL)
    original = Image.open(image_path).convert('L')
    
    # Apply transforms
    transformed = transform(image=image_gray)
    processed_image = transformed['image'].unsqueeze(0)
    
    return processed_image, original


def generate_gradcam_heatmap(model, image_tensor, original_image, target_layer_name='backbone'):
    """Generate GradCAM heatmap overlay"""
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
    
    return prediction, confidence


def create_detailed_report(original_image, overlay, prediction, confidence, region_info):
    """Create detailed visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    original_array = np.array(original_image.resize((224, 224)))
    axes[0].imshow(original_array, cmap='gray')
    axes[0].set_title('Original X-Ray')
    axes[0].axis('off')
    
    # Analysis results
    axes[1].axis('off')
    class_name = 'PNEUMONIA' if prediction == 1 else 'NORMAL'
    
    result_text = f"Analysis Result\n\n"
    result_text += f"Classification: {class_name}\n"
    result_text += f"Confidence: {confidence*100:.1f}%\n"
    result_text += f"Affected Area: {region_info.get('affected_percentage', 0):.1f}%"
    
    axes[1].text(0.1, 0.5, result_text, fontsize=13, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig


def analyze_cam_regions(cam, threshold=0.6):
    """Analyze regions"""
    binary_mask = (cam > threshold).astype(np.uint8)
    affected_percentage = (binary_mask.sum() / binary_mask.size) * 100
    
    region_info = {
        'affected_percentage': affected_percentage,
        'num_regions': 1,
        'max_attention': cam.max(),
        'mean_attention': cam.mean(),
        'std_attention': cam.std()
    }
    
    return affected_percentage, region_info