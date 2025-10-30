import os
import torch
import gradio as gr
import cv2
import numpy as np
import tempfile
import functools
from medical_pretrained_model import MedicalPretrainedModel
from improved_visualization import load_and_preprocess_image, analyze_cam_regions
from image_classifier import ImageClassifier
import matplotlib.pyplot as plt
import json


@functools.lru_cache(maxsize=1)
def get_medical_model(model_path='models/medical_best_model.pth'):
    """Load and cache the medical model"""
    try:
        print(f"Loading medical model from: {model_path}")
        
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = MedicalPretrainedModel(model_name='densenet121-res224-all', num_classes=2)
        
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print("Medical model loaded successfully")
        return model, device
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise gr.Error(f"Error loading model: {str(e)}")


@functools.lru_cache(maxsize=1)
def get_yolo_classifier():
    """Load and cache YOLO classifier"""
    print("Loading YOLO classifier...")
    classifier = ImageClassifier()
    print("YOLO classifier loaded")
    return classifier


def convert_to_grayscale(image_input):
    """Convert image to grayscale for model compatibility"""
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
            image_input = np.expand_dims(image_input, axis=-1)
    return image_input


def predict_xray(image_input, model_path='models/medical_best_model.pth'):
    """Predict chest X-ray"""
    temp_file_path = None
    
    try:
        if isinstance(image_input, np.ndarray):
            temp_fd, temp_file_path = tempfile.mkstemp(suffix='.jpg')
            os.close(temp_fd)
            
            image_to_save = convert_to_grayscale(image_input.copy())
            
            if len(image_to_save.shape) == 3:
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
            else:
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_GRAY2BGR)
            
            cv2.imwrite(temp_file_path, image_to_save)
            image_path = temp_file_path
        else:
            image_path = image_input
        
        if not os.path.exists(image_path):
            raise gr.Error("Image file not found")
        
        model, device = get_medical_model(model_path)
        
        image_tensor, original_image = load_and_preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            prediction = outputs.argmax(dim=1).item()
            confidence = probs[0, prediction].item()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        original_array = np.array(original_image.resize((224, 224)))
        axes[0].imshow(original_array, cmap='gray')
        axes[0].set_title('Original X-Ray')
        axes[0].axis('off')
        
        axes[1].axis('off')
        class_name = 'PNEUMONIA' if prediction == 1 else 'NORMAL'
        
        result_text = f"Classification\n\n"
        result_text += f"Result: {class_name}\n"
        result_text += f"Confidence: {confidence*100:.1f}%"
        
        axes[1].text(0.1, 0.5, result_text, fontsize=13, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Report
        report = f"""ANALYSIS RESULT

Classification: {class_name}
Confidence: {confidence*100:.1f}%

"""
        
        if prediction == 1:
            if confidence > 0.90:
                report += "High confidence: Pneumonia indicators present."
            else:
                report += "Moderate confidence: Possible pneumonia indicators."
        else:
            if confidence > 0.90:
                report += "High confidence: No significant abnormalities detected."
            else:
                report += "Moderate confidence: Likely normal appearance."
        
        return fig, report, float(confidence)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error during prediction: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


def predict_object(image_input):
    """Predict general objects"""
    temp_file_path = None
    
    try:
        if isinstance(image_input, np.ndarray):
            temp_fd, temp_file_path = tempfile.mkstemp(suffix='.jpg')
            os.close(temp_fd)
            
            if len(image_input.shape) == 3:
                image_to_save = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
            else:
                image_to_save = image_input
            
            cv2.imwrite(temp_file_path, image_to_save)
            image_path = temp_file_path
        else:
            image_path = image_input
        
        classifier = get_yolo_classifier()
        results = classifier.model(image_path, conf=0.25, augment=True)
        detections = results[0].boxes
        
        fig, ax = plt.subplots(figsize=(14, 10))
        img = results[0].plot(conf=True, labels=True, boxes=True, line_width=3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        detection_count = len(detections)
        ax.imshow(img)
        ax.set_title(f"Object Detection - {detection_count} object(s) detected", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if len(detections) > 0:
            detection_list = []
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = classifier.model.names[cls_id]
                detection_list.append(f"  - {label}: {conf*100:.1f}%")
            
            result_text = "OBJECTS DETECTED:\n\n" + "\n".join(detection_list)
            result_text += f"\n\nTotal: {detection_count}"
        else:
            result_text = "No objects detected"
        
        return fig, result_text, 0.0
    
    except Exception as e:
        print(f"Error: {e}")
        raise gr.Error(f"Error: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


def predict(image_input):
    """Main prediction function"""
    if image_input is None:
        raise gr.Error("Please upload an image or capture from camera")
    
    classifier = get_yolo_classifier()
    
    if isinstance(image_input, np.ndarray):
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        img_save = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if len(image_input.shape) == 3 else image_input
        cv2.imwrite(temp_path, img_save)
        check_path = temp_path
    else:
        check_path = image_input
    
    is_xray, _ = classifier.classify_image(check_path)
    
    if isinstance(image_input, np.ndarray):
        try:
            os.unlink(check_path)
        except:
            pass
    
    if is_xray:
        return predict_xray(image_input)
    else:
        return predict_object(image_input)


def create_interface():
    """Create Gradio interface"""
    
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 600 !important;
    }
    .markdown-text {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    button {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 500 !important;
    }
    """
    
    with gr.Blocks(title="Hybrid AI System for Detecting Anomalies in Chest X-rays", theme=gr.themes.Soft(), css=custom_css) as iface:
        
        gr.Markdown("# **Hybrid AI System for Detecting Anomalies in Chest X-rays**")
        
        gr.Markdown("""
## Analysis System

Analyze images for medical X-ray screening and general object detection

### For Chest X-rays:
- Classification: Normal or Pneumonia detection
- Confidence scores and detailed analysis
- Support for medical screening workflows

### For Other Images:
- Object detection using advanced computer vision
- Detailed detection reports with confidence levels
- Multi-object identification capability

---

### Instructions:
1. Upload an image or capture from camera
2. Click Analyze to process
3. Review results and confidence scores

### What This System Uses:
- **X-Ray Model**: DenseNet121 with medical pretraining
- **Training Data**: 200,000+ chest X-rays from multiple datasets
- **Object Detection**: YOLOv8 for general images
- **Purpose**: Screening support and analysis tool
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Input")
                
                with gr.Tabs():
                    with gr.TabItem("Upload"):
                        input_image = gr.Image(
                            label="Upload Image",
                            type="filepath",
                            sources=["upload"]
                        )
                    
                    with gr.TabItem("Camera"):
                        camera_image = gr.Image(
                            label="Capture from Camera",
                            type="numpy",
                            sources=["webcam"]
                        )
                
                analyze_btn = gr.Button(
                    "Analyze",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## Results")
                
                output_plot = gr.Plot(label="Visualization")
                diagnosis = gr.Textbox(
                    label="Analysis Report",
                    lines=15,
                    max_lines=20
                )
                confidence = gr.Number(label="Confidence Score")
        
        gr.Markdown("""
---

## System Information

**Model Architecture:**
- Medical X-Ray: DenseNet121 (pretrained)
- General Detection: YOLOv8
- Image Processing: Albumentations with CLAHE enhancement

**Test Performance:**
- X-Ray Accuracy: 87.84%
- Processing: Optimized with model caching

**Medical Disclaimer:**
This system is provided for educational and research purposes only. Results must be verified by qualified professionals. Not a medical device. Always consult healthcare professionals for medical decisions.

---

<div style="text-align: center; padding: 20px; color: #666;">
            <p style="font-size: 14px; margin: 0;">Developed by <strong>Aditya Rai</strong></p>
        </div>
        """)
        
        def handle_analysis(upload_img, camera_img):
            active_input = camera_img if camera_img is not None else upload_img
            
            if active_input is None:
                raise gr.Error("Please upload an image or capture from camera")
            
            return predict(active_input)
        
        analyze_btn.click(
            fn=handle_analysis,
            inputs=[input_image, camera_image],
            outputs=[output_plot, diagnosis, confidence]
        )
    
    return iface


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Medical X-Ray Analysis System")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else ('CUDA' if torch.cuda.is_available() else 'CPU')}")
    print("="*60 + "\n")
    
    iface = create_interface()
    iface.launch()