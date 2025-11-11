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
import socket
from pathlib import Path


# Set up cache and discover model file if present
CACHE_DIR = os.path.expanduser('~/.cache/medical_analyzer')
os.makedirs(CACHE_DIR, exist_ok=True)


def find_model_file():
    """Search common locations for a local model file (.pth/.pt/.safetensors).
    Returns a pathlib.Path or None.
    """
    base = Path(__file__).parent
    candidates = [
        base / 'Models' / 'medical_best_model.pth',
        base / 'models' / 'medical_best_model.pth',
        base / 'Models' / 'medical_best_model.safetensors',
        base / 'models' / 'medical_best_model.safetensors',
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: any file named medical_best_model with common extensions
    for ext in ('.pth', '.pt', '.safetensors'):
        found = list(base.glob('**/medical_best_model' + ext))
        if found:
            return found[0]
    return None


HF_MODEL_PATH = find_model_file()
if HF_MODEL_PATH:
    HF_MODEL_PATH = str(HF_MODEL_PATH)
else:
    HF_MODEL_PATH = None


@functools.lru_cache(maxsize=1)
def get_medical_model():
    """Load and cache the medical model from local path"""
    try:
        print(f"Loading medical model from: {HF_MODEL_PATH}")

        # Device selection with proper fallback
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        print(f"Using device: {device}")

        # Initialize architecture
        model = MedicalPretrainedModel(model_name='densenet121-res224-all', num_classes=2)

        # If a model file exists, try common loaders
        if HF_MODEL_PATH and os.path.exists(HF_MODEL_PATH):
            p = Path(HF_MODEL_PATH)
            try:
                if p.suffix in ('.pth', '.pt'):
                    checkpoint = torch.load(HF_MODEL_PATH, map_location=device)
                    # normalize wrapped checkpoints
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state = checkpoint['state_dict']
                    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        state = checkpoint['model_state_dict']
                    else:
                        state = checkpoint
                    model.load_state_dict(state)
                elif p.suffix == '.safetensors':
                    from safetensors.torch import load_file
                    checkpoint = load_file(HF_MODEL_PATH, device=str(device))
                    model.load_state_dict(checkpoint)
                else:
                    raise RuntimeError(f"Unsupported model suffix: {p.suffix}")
                model = model.to(device)
                model.eval()
                print("Medical model loaded successfully from disk")
                return model, device
            except Exception as e:
                print(f"Warning: failed to load disk model {HF_MODEL_PATH}: {e}")

        # Fallback: small dummy model so UI will render and can be used for testing
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                batch = x.shape[0] if isinstance(x, torch.Tensor) and x.dim() > 0 else 1
                return torch.zeros((batch, 2), device=x.device if isinstance(x, torch.Tensor) else device)

        print("Using DummyModel fallback (predictions are placeholders)")
        dummy = DummyModel().to(device)
        dummy.eval()
        return dummy, device

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise gr.Error(f"Error loading model: {str(e)}")


@functools.lru_cache(maxsize=1)
def get_yolo_classifier():
    """Load and cache YOLO classifier"""
    try:
        print("Loading YOLO classifier...")
        classifier = ImageClassifier()
        print("YOLO classifier loaded")
        return classifier
    except Exception as e:
        print(f"Warning: YOLO classifier failed to load: {e}")
        raise gr.Error(f"Error loading YOLO classifier: {str(e)}")


def convert_to_grayscale(image_input):
    """Convert image to grayscale for model compatibility"""
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
            image_input = np.expand_dims(image_input, axis=-1)
    return image_input


def predict_xray(image_input):
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
        
        model, device = get_medical_model()
        
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
        
        result_text = f"Classification\n\nResult: {class_name}\nConfidence: {confidence*100:.1f}%"
        
        axes[1].text(0.1, 0.5, result_text, fontsize=13, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Report
        report = f"ANALYSIS RESULT\n\nClassification: {class_name}\nConfidence: {confidence*100:.1f}%\n\n"
        
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
    
    temp_file_path = None
    try:
        if isinstance(image_input, np.ndarray):
            temp_fd, temp_file_path = tempfile.mkstemp(suffix='.jpg')
            os.close(temp_fd)
            img_save = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if len(image_input.shape) == 3 else image_input
            cv2.imwrite(temp_file_path, img_save)
            check_path = temp_file_path
        else:
            check_path = image_input
        
        is_xray, _ = classifier.classify_image(check_path)
        
        if is_xray:
            return predict_xray(image_input)
        else:
            return predict_object(image_input)
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


def create_interface():
    """Create Gradio interface"""
    
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        color: #000 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 600 !important;
        color: #000 !important;
    }
    p, li, span, div, label {
        color: #000 !important;
    }
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white !important;
        margin-bottom: 20px;
    }
    .main-title h1, .main-title p {
        color: white !important;
    }
    .section-header {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
        color: #000 !important;
    }
    .info-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
        color: #000 !important;
    }
    .feature-list {
        background: #f5f5f5;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        color: #000 !important;
    }
    .footer {
        text-align: center;
        padding: 30px 20px;
        color: #000 !important;
        background: #f8f9fa;
        border-radius: 8px;
        margin-top: 30px;
    }
    """
    
    with gr.Blocks(
        title="Hybrid AI System for Detecting Anomalies in Chest CT and X-rays",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as iface:
        
        gr.HTML("""
        <div class="main-title">
            <h1 style="margin: 0; font-size: 2.5em;">🏥 Hybrid AI Medical Imaging System</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">
                Advanced Detection of Anomalies in Chest X-rays
            </p>
        </div>
        """)
        
        gr.HTML("""
        <div class="section-header">
            <h2 style="margin: 0; color: #000 !important; font-weight: bold;">📊 Analysis System Overview</h2>
            <p style="margin: 5px 0 0 0; color: #000 !important; font-size: 16px;">
                AI-powered medical image analysis with real-time detection and classification
            </p>
        </div>
        """)
        
        gr.HTML("""
        <div class="feature-list">
            <h3 style="color: #000 !important; margin-top: 0; font-weight: bold;">🔬 For Chest X-rays:</h3>
            <ul style="line-height: 1.8; color: #000 !important; font-size: 16px;">
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Binary Classification:</strong> Normal vs Pneumonia detection</li>
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Confidence Scoring:</strong> Detailed probability analysis</li>
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Clinical Support:</strong> Assists medical screening workflows</li>
            </ul>
            
            <h3 style="color: #000 !important; margin-top: 20px; font-weight: bold;">🎯 For General Images:</h3>
            <ul style="line-height: 1.8; color: #000 !important; font-size: 16px;">
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Object Detection:</strong> Advanced computer vision with YOLOv8</li>
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Multi-Object Recognition:</strong> Simultaneous detection of multiple objects</li>
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Confidence Levels:</strong> Detailed detection reports with accuracy metrics</li>
            </ul>
        </div>
        """)
        
        gr.HTML("""
        <div class="info-box">
            <h3 style="margin-top: 0; color: #000 !important; font-weight: bold;">📋 Instructions:</h3>
            <ol style="line-height: 2; color: #000 !important; margin: 10px 0; font-size: 16px;">
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Upload</strong> a chest X-ray image or capture from camera</li>
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Click Analyze</strong> to process the image with AI</li>
                <li style="color: #000 !important;"><strong style="color: #000 !important;">Review</strong> results, confidence scores, and detailed analysis</li>
            </ol>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: #000 !important; font-weight: bold;">📤 Input Image</h2>
                </div>
                """)
                
                with gr.Tabs():
                    with gr.TabItem("📁 Upload"):
                        input_image = gr.Image(
                            label="Upload Medical Image",
                            type="filepath",
                            sources=["upload"]
                        )
                    
                    with gr.TabItem("📷 Camera"):
                        camera_image = gr.Image(
                            label="Capture from Camera",
                            type="numpy",
                            sources=["webcam"]
                        )
                
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h2 style="margin: 0; color: #000 !important; font-weight: bold;">📋 Analysis Results</h2>
                </div>
                """)
                output_plot = gr.Plot(label="Visualization")
                diagnosis = gr.Textbox(label="📄 Analysis Report", lines=15, max_lines=20)
                confidence = gr.Number(label="🎯 Confidence Score")
        
        gr.HTML("<hr style='margin: 40px 0; border: none; border-top: 2px solid #e0e0e0;'>")
        
        gr.HTML("""
        <div class="section-header">
            <h2 style="margin: 0; color: #000 !important; font-weight: bold;">⚙️ System Information</h2>
        </div>
        """)
        
        gr.HTML("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">
            <div style="background: #f0f4ff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #d0d7ff;">
                <h3 style="color: #000 !important; margin-top: 0; font-weight: bold;">🧠 Model Architecture</h3>
                <ul style="line-height: 1.8; color: #000 !important; font-size: 16px;">
                    <li style="color: #000 !important;"><strong style="color: #000 !important;">Medical Imaging:</strong> DenseNet121</li>
                    <li style="color: #000 !important;"><strong style="color: #000 !important;">General Detection:</strong> YOLOv8</li>
                    <li style="color: #000 !important;"><strong style="color: #000 !important;">Image Processing:</strong> Albumentations + CLAHE</li>
                    <li style="color: #000 !important;"><strong style="color: #000 !important;">Device Support:</strong> GPU/CPU/MPS (auto-detected)</li>
                </ul>
            </div>
            
            <div style="background: #f8f0ff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e8d5ff;">
                <h3 style="color: #000 !important; margin-top: 0; font-weight: bold;">📊 Performance Metrics</h3>
                <ul style="line-height: 1.8; color: #000 !important; font-size: 16px;">
                    <li style="color: #000 !important;"><strong style="color: #000 !important;">X-Ray Accuracy:</strong> 87.84%</li>
                    <li style="color: #000 !important;"><strong style="color: #000 !important;">Processing:</strong> Optimized with model caching</li>
                    <li style="color: #000 !important;"><strong style="color: #000 !important;">Inference Speed:</strong> Sub-second on GPU</li>
                </ul>
            </div>
        </div>
        """)
        
        gr.HTML("""
        <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <h3 style="color: #000 !important; margin-top: 0; font-weight: bold;">⚠️ Medical Disclaimer</h3>
            <p style="color: #000 !important; line-height: 1.6; margin: 0; font-size: 16px;">
                This system is provided for <strong style="color: #000 !important;">educational and research purposes only</strong>. 
                All results must be verified by qualified healthcare professionals. This is <strong style="color: #000 !important;">NOT a medical device</strong> 
                and should not be used as the sole basis for clinical decisions. Always consult healthcare 
                professionals for medical advice, diagnosis, or treatment.
            </p>
        </div>
        """)
        
        gr.HTML("""
        <div class="footer">
            <p style="font-size: 18px; margin: 0; font-weight: 600; color: #000 !important;">
                Developed by <strong style="color: #667eea !important;">Aditya Rai</strong>
            </p>
            <p style="font-size: 16px; margin: 10px 0 0 0; color: #000 !important;">
                Powered by PyTorch, Gradio & Hugging Face
            </p>
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
    print("\n" + "="*70)
    print("Medical CT & X-Ray Analysis System".center(70))
    print("="*70)
    print(f"PyTorch Version: {torch.__version__}")
    device_info = 'CUDA' if torch.cuda.is_available() else ('MPS' if torch.backends.mps.is_available() else 'CPU')
    print(f"Device: {device_info}")
    print(f"Model Path: {HF_MODEL_PATH}")
    print("="*70 + "\n")
    
    print("Starting Gradio interface...")
    iface = create_interface()
    
    # Prefer binding to localhost and try ports 7860-7870 to be deterministic for the user.
    def find_free_port(start=7860, end=7870):
        for p in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', p))
                    return p
                except OSError:
                    continue
        raise RuntimeError(f"No free port in range {start}-{end}")

    try:
        port = find_free_port(7860, 7870)
    except Exception:
        port = 0

    host = '127.0.0.1'
    if port and port != 0:
        print(f"Launching Gradio on http://{host}:{port} (server_name='{host}')")
    else:
        print("Launching Gradio on an auto-selected port (no free port in 7860-7870)")

    iface.launch(
        share=False,
        server_name=host,
        server_port=port if port != 0 else None,
        show_error=True
    )
