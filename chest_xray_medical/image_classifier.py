from ultralytics import YOLO
import torch
import numpy as np
import cv2

class ImageClassifier:
    def __init__(self):
        # Load YOLOv8 model with best weights for object detection
        try:
            self.model = YOLO('yolov8x.pt')  # Using YOLOv8x which is larger and more accurate
        except Exception as e:
            print(f"Error loading YOLOv8x: {e}")
            print("Falling back to smaller model...")
            self.model = YOLO('yolov8n.pt')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Configure model for better detection
        self.model.conf = 0.15  # Lower confidence threshold for better detection
        self.model.iou = 0.3   # Lower IOU threshold to detect more overlapping objects
    
    def classify_xray_type(self, img):
        """
        Determine the type of X-ray based on image characteristics.
        """
        from scipy import stats
        
        try:
            height, width = img.shape
            aspect_ratio = width / height
        
            # Calculate region statistics
            top_region = img[:height//3, :]
            middle_region = img[height//3:2*height//3, :]
            bottom_region = img[2*height//3:, :]
            
            region_means = [np.mean(r) for r in [top_region, middle_region, bottom_region]]
            region_stds = [np.std(r) for r in [top_region, middle_region, bottom_region]]
            
            # Calculate histogram features
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            hist_entropy = stats.entropy(hist_norm + 1e-7)
            
            # Decision logic for X-ray types
            if 0.7 <= aspect_ratio <= 1.3:  # Square-ish
                if region_means[0] > region_means[1] and hist_entropy < 4.5:
                    return "Dental X-ray"
            
            if 1.3 <= aspect_ratio <= 1.9:  # Portrait
                if region_stds[1] > region_stds[0] and region_stds[1] > region_stds[2]:
                    return "Chest X-ray"
                elif region_stds[2] > region_stds[1]:
                    return "Abdominal X-ray"
            
            if aspect_ratio > 1.7:  # Very tall/wide
                if width > height:
                    return "Full Body X-ray"
                else:
                    return "Spine X-ray"
                    
            # Check for bone X-rays and extremities
            edges = cv2.Canny(img, 100, 200)
            edge_density = np.count_nonzero(edges) / (height * width)
            high_contrast = max(region_stds) > 60

            center = img[height//4:3*height//4, width//4:3*width//4]
            center_edges = cv2.Canny(center, 100, 200)
            center_edge_density = np.count_nonzero(center_edges) / (center.shape[0] * center.shape[1])

            if edge_density > 0.12 and center_edge_density > 0.15 and high_contrast:
                return "Hand X-ray"
            if edge_density > 0.1 and high_contrast:
                return "Bone X-ray"
                
            return "General X-ray"
            
        except Exception as e:
            print(f"Error in X-ray type classification: {e}")
            return "Unspecified X-ray"

    def classify_image(self, image_path):
        """
        Classify an image and determine if it's a chest X-ray or something else.
        Returns: tuple (is_xray, detected_objects)
        """
        try:
            # Load the image in both color and grayscale
            if isinstance(image_path, np.ndarray):
                # Direct numpy array input (from camera)
                img_color = image_path.copy()
                if len(image_path.shape) == 3:
                    img_gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = image_path
            else:
                # File path input
                img_color = cv2.imread(str(image_path))
                img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            if img_color is None or img_gray is None:
                print(f"Error loading image. Image path type: {type(image_path)}")
                return False, ["Error loading image"]
                
            print(f"Image loaded - Color shape: {img_color.shape}, Gray shape: {img_gray.shape}")
            
            # STEP 1: Run YOLO detection first (most reliable)
            print("Running YOLO detection...")
            self.model.conf = 0.1  # Lower confidence for better detection
            results = self.model(img_color, verbose=False, augment=True)
            
            # Collect all detections
            all_detections = []
            has_person = False
            has_common_objects = False
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    conf = box.conf.item()
                    label = result.names[class_id]
                    
                    if conf > 0.15:  # Only consider confident detections
                        all_detections.append((label, conf))
                        
                        # Check for persons
                        if label.lower() == "person":
                            has_person = True
                            
                        # Check for common non-medical objects
                        common_objects = ['person', 'car', 'chair', 'table', 'bottle', 'cup', 
                                        'laptop', 'mouse', 'keyboard', 'cell phone', 'book',
                                        'clock', 'vase', 'scissors', 'teddy bear', 'dog', 'cat',
                                        'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                                        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                                        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                                        'sports ball', 'kite', 'baseball bat', 'skateboard',
                                        'surfboard', 'tennis racket', 'wine glass', 'fork',
                                        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                                        'cake', 'couch', 'potted plant', 'bed', 'dining table',
                                        'toilet', 'tv', 'remote', 'microwave', 'oven', 'toaster',
                                        'sink', 'refrigerator', 'hair drier', 'toothbrush']
                        
                        if label.lower() in common_objects:
                            has_common_objects = True
            
            print(f"Detections: {len(all_detections)} objects found")
            print(f"Has person: {has_person}, Has common objects: {has_common_objects}")
            
            # STEP 2: Check if image is colorful (not X-ray)
            if len(img_color.shape) == 3 and img_color.shape[2] == 3:
                # Calculate color variance across channels
                color_std = np.std(img_color, axis=2).mean()
                
                # Calculate difference between color channels
                b, g, r = cv2.split(img_color)
                channel_diff = np.abs(r.astype(float) - b.astype(float)).mean()
                
                is_colorful = (color_std > 25) or (channel_diff > 20)
                print(f"Color analysis - std: {color_std:.2f}, channel_diff: {channel_diff:.2f}, is_colorful: {is_colorful}")
            else:
                is_colorful = False
                print("Image is already grayscale")
            
            # STEP 3: If person or common objects detected, check if it's actually an X-ray
            # X-rays can sometimes trigger "person" detection due to body silhouette
            if has_person or has_common_objects:
                print("Person/objects detected - checking if it's an X-ray...")
                
                # Additional checks for X-rays that might have person detections
                # X-rays have very specific characteristics even if YOLO detects a person
                
                # Check if image looks like medical imaging
                mean_intensity = np.mean(img_gray)
                std_intensity = np.std(img_gray)
                
                # Calculate edge density (X-rays have characteristic bone edges)
                edges = cv2.Canny(img_gray, 50, 150)
                edge_density = np.count_nonzero(edges) / (img_gray.shape[0] * img_gray.shape[1])
                
                # Check histogram characteristics
                hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
                hist_peak = np.argmax(hist)
                
                # X-rays typically have:
                # 1. Dark background (low mean if inverted, or specific range)
                # 2. High contrast (high std)
                # 3. Specific edge patterns (bone structures)
                # 4. Grayscale or very low color variance
                
                looks_like_xray = (
                    (std_intensity > 45 and 20 < mean_intensity < 180) and  # Contrast and intensity
                    (not is_colorful) and  # Must be grayscale
                    (edge_density > 0.05 or std_intensity > 55)  # Has structure or very high contrast
                )
                
                if looks_like_xray:
                    print("Override: Detected as X-ray despite person detection (medical imaging characteristics)")
                    xray_type = self.classify_xray_type(img_gray)
                    return True, [f"{xray_type} detected"]
                
                print("Confirmed as NON-XRAY (detected person/objects)")
                
                # Format detected objects
                all_detections.sort(key=lambda x: x[1], reverse=True)
                seen_labels = set()
                detected_objects = []
                
                for label, conf in all_detections:
                    if label.lower() not in seen_labels:
                        seen_labels.add(label.lower())
                        detected_objects.append(f"{label} (Confidence: {conf*100:.1f}%)")
                
                if not detected_objects:
                    detected_objects = ["Objects detected but with low confidence"]
                
                return False, detected_objects
            
            # STEP 4: If image is colorful, it's NOT an X-ray
            if is_colorful:
                print("âœ“ Classified as NON-XRAY (colorful image)")
                if all_detections:
                    all_detections.sort(key=lambda x: x[1], reverse=True)
                    detected_objects = [f"{label} (Confidence: {conf*100:.1f}%)" 
                                      for label, conf in all_detections[:5]]
                else:
                    detected_objects = ["Colorful image - not a medical X-ray"]
                return False, detected_objects
            
            # STEP 5: Check if it could be an X-ray based on image characteristics
            mean_intensity = np.mean(img_gray)
            std_intensity = np.std(img_gray)
            
            print(f"Grayscale analysis - mean: {mean_intensity:.2f}, std: {std_intensity:.2f}")
            
            # Calculate additional X-ray indicators
            edges = cv2.Canny(img_gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (img_gray.shape[0] * img_gray.shape[1])
            
            print(f"Edge density: {edge_density:.4f}")
            
            # X-ray images typically have:
            # 1. High contrast (high std deviation)
            # 2. Moderate to low mean intensity (dark background, bright bones)
            # 3. Specific intensity range
            # 4. Characteristic edge patterns from bone/tissue boundaries
            is_xray_characteristics = (
                std_intensity > 40 and  # High contrast (lowered from 45)
                15 < mean_intensity < 200 and  # Broader intensity range
                not is_colorful and  # Must be grayscale-like
                (edge_density > 0.04 or std_intensity > 50)  # Has structure or very high contrast
            )
            
            if is_xray_characteristics:
                print("âœ“ Classified as XRAY (medical image characteristics)")
                xray_type = self.classify_xray_type(img_gray)
                return True, [f"{xray_type} detected"]
            
            # STEP 6: Default to non-X-ray with object detection results
            print("âœ“ Classified as NON-XRAY (default - no X-ray characteristics)")
            if all_detections:
                all_detections.sort(key=lambda x: x[1], reverse=True)
                detected_objects = [f"{label} (Confidence: {conf*100:.1f}%)" 
                                  for label, conf in all_detections[:5]]
            else:
                detected_objects = ["No specific objects detected with high confidence"]
            
            return False, detected_objects
            
        except Exception as e:
            print(f"Error in image classification: {e}")
            import traceback
            traceback.print_exc()
            return False, [f"Error during classification: {str(e)}"]