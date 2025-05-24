import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from torchvision import transforms, models
import requests
import time
import sys
from model_loader import prepare_models

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for Flutter integration

# Load the trained model
MODEL_PATH = os.path.join('weights', 'best.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
yolo_model = None
CLASS_NAMES = []

# Image preprocessing transformer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Confidence thresholds
CLASSIFICATION_THRESHOLD = 0.75  # 75% for classification confidence
DETECTION_THRESHOLD = 0.4  # 40% for detection confidence

# YOLOv5 detection classes that could be road signs
ROAD_SIGN_CLASSES = [
    'traffic light', 'stop sign', 'parking meter', 'fire hydrant',
    'car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle',
    'traffic sign', 'street sign', 'pole', 'signboard'
]
# Path to custom YOLOv5 weights if available
CUSTOM_YOLO_PATH = os.path.join('weights', 'yolov5_traffic_signs.pt')

def create_model(num_classes):
    """Create a MobileNetV3 model for classification"""
    # Load a pre-trained MobileNetV3 Small model
    model = models.mobilenet_v3_small(weights=None)
    
    # Replace the classifier with a new one for our number of classes
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    return model

def load_yolo():
    """Load YOLOv5 model for object detection, prioritizing traffic sign detection"""
    global yolo_model
    try:
        # First try to load a specialized traffic sign detection model if available
        if os.path.exists(CUSTOM_YOLO_PATH):
            print(f"Loading specialized traffic sign detection model from {CUSTOM_YOLO_PATH}")
            # Custom model loading
            yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=CUSTOM_YOLO_PATH)
            yolo_model.to(DEVICE)
            yolo_model.eval()
            print("Specialized traffic sign detection model loaded successfully")
            return yolo_model
        
        # If no specialized model, try to load YOLOv5m (medium) for better accuracy than small
        print("No specialized model found, loading YOLOv5m...")
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        
        # Configure model for better small object detection (road signs are often small in images)
        yolo_model.conf = 0.25  # Lower confidence threshold for detection
        yolo_model.iou = 0.45   # IOU threshold for NMS
        yolo_model.classes = None  # Detect all classes
        yolo_model.max_det = 100  # Increase maximum detections
        
        yolo_model.to(DEVICE)
        yolo_model.eval()
        print("YOLOv5m model loaded and optimized for small object detection")
        return yolo_model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        try:
            # Fallback to smaller model which might be more compatible
            print("Falling back to YOLOv5s...")
            yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            yolo_model.conf = 0.25  # Lower confidence threshold
            yolo_model.to(DEVICE)
            yolo_model.eval()
            print("YOLOv5s model loaded successfully as fallback")
            return yolo_model
        except Exception as e2:
            print(f"Error loading fallback YOLOv5 model: {e2}")
            print("Using YOLOv5 API fallback for detection")
            return None

def load_model():
    global model, yolo_model, CLASS_NAMES
    # Load class names
    with open('classes.txt', 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    
    # Load classification model
    num_classes = len(CLASS_NAMES)
    model = create_model(num_classes)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Classification model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Using untrained model.")
    
    # Load YOLOv5 detection model
    yolo_model = load_yolo()
    
    return model

def enhance_image_for_detection(img):
    """Enhance the image to make road signs more visible"""
    try:
        # Convert PIL Image to OpenCV format
        img_cv = np.array(img)
        img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR for OpenCV
        
        # Create a copy of the original image for enhancement
        enhanced = img_cv.copy()
        
        # Apply contrast enhancement
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Convert back to PIL Image
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        return enhanced_pil
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return img  # Return original if enhancement fails

def detect_road_signs(img):
    """Detect potential road signs in the image using YOLOv5"""
    try:
        if yolo_model is not None:
            # Create multiple versions of the image for better detection
            images = [
                img,  # Original image
                enhance_image_for_detection(img)  # Enhanced image
            ]
            
            all_detections = []
            
            # Run detection on each image version
            for img_version in images:
                # Use the loaded YOLOv5 model
                results = yolo_model(img_version)
                detections = results.pandas().xyxy[0]  # Get detection results as DataFrame
                
                # Filter detections for potential road signs
                for _, detection in detections.iterrows():
                    if detection['confidence'] >= DETECTION_THRESHOLD:
                        # Check if it's a known road sign class or has high confidence
                        if detection['name'] in ROAD_SIGN_CLASSES or detection['confidence'] > 0.6:
                            # Expand the bounding box slightly to ensure the whole sign is captured
                            xmin = max(0, detection['xmin'] - 5)
                            ymin = max(0, detection['ymin'] - 5)
                            xmax = min(img.width, detection['xmax'] + 5)
                            ymax = min(img.height, detection['ymax'] + 5)
                            
                            all_detections.append({
                                'bbox': [xmin, ymin, xmax, ymax],
                                'confidence': detection['confidence'],
                                'class': detection['name']
                            })
            
            # Merge overlapping detections (non-maximum suppression)
            final_detections = []
            all_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            for detection in all_detections:
                should_add = True
                for final_det in final_detections:
                    # Calculate IoU (Intersection over Union)
                    bbox1 = detection['bbox']
                    bbox2 = final_det['bbox']
                    
                    # Calculate intersection area
                    x_left = max(bbox1[0], bbox2[0])
                    y_top = max(bbox1[1], bbox2[1])
                    x_right = min(bbox1[2], bbox2[2])
                    y_bottom = min(bbox1[3], bbox2[3])
                    
                    if x_right < x_left or y_bottom < y_top:
                        intersection_area = 0
                    else:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    
                    # Calculate union area
                    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    union_area = bbox1_area + bbox2_area - intersection_area
                    
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    if iou > 0.5:  # If there's significant overlap
                        should_add = False
                        break
                
                if should_add:
                    final_detections.append(detection)
            
            # If no road signs detected, try a more aggressive approach with lower threshold
            if not final_detections:
                # Use a lower confidence threshold for a second pass
                results = yolo_model(enhance_image_for_detection(img))
                detections = results.pandas().xyxy[0]
                
                for _, detection in detections.iterrows():
                    if detection['confidence'] >= 0.2:  # Lower threshold
                        # Expand the bounding box more aggressively
                        xmin = max(0, detection['xmin'] - 10)
                        ymin = max(0, detection['ymin'] - 10)
                        xmax = min(img.width, detection['xmax'] + 10)
                        ymax = min(img.height, detection['ymax'] + 10)
                        
                        final_detections.append({
                            'bbox': [xmin, ymin, xmax, ymax],
                            'confidence': detection['confidence'],
                            'class': detection['name']
                        })
            
            return final_detections
        else:
            # Fallback to using a simple color-based detection approach
            # This is a very basic approach that looks for common road sign colors
            img_cv = np.array(img)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common road sign colors (red, blue, yellow)
            # Red range (wraps around in HSV)  
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # Blue range
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
            
            # Yellow range
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # Create masks
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Combine masks
            combined_mask = cv2.bitwise_or(mask_red, mask_blue)
            combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            potential_signs = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    # Most road signs have aspect ratios close to 1
                    if 0.5 <= aspect_ratio <= 2.0:
                        potential_signs.append({
                            'bbox': [float(x), float(y), float(x + w), float(y + h)],
                            'confidence': 0.3,  # Default confidence for color-based detection
                            'class': 'potential_sign'
                        })
            
            return potential_signs
            
    except Exception as e:
        print(f"Error in detection: {e}")
        return []

def crop_and_classify(img, bbox):
    """Crop a road sign from the image and classify it"""
    try:
        # Ensure the image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Crop the image to the bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cropped_img = img.crop((x1, y1, x2, y2))
        
        # Apply transforms for classification
        img_tensor = transform(cropped_img)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                confidence = prob.item()
                class_id = idx.item()
                
                # Only include predictions with confidence above threshold
                if confidence >= CLASSIFICATION_THRESHOLD or i == 0:  # Always include top prediction
                    predictions.append({
                        'class_id': class_id,
                        'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'Unknown',
                        'confidence': confidence
                    })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Filter out predictions below threshold
            filtered_predictions = [p for p in predictions if p['confidence'] >= CLASSIFICATION_THRESHOLD]
            
            return {
                'predictions': filtered_predictions,
                'has_prediction': len(filtered_predictions) > 0,
                'top_prediction': predictions[0] if predictions else None,
                'bbox': bbox
            }
    except Exception as e:
        print(f"Error in classification: {e}")
        return None

def process_image(img):
    """Two-stage process: first detect road signs, then classify each detection"""
    try:
        # Ensure the image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Stage 1: Detect potential road signs in the image
        detections = detect_road_signs(img)
        
        if not detections:
            # If no road signs detected, perform direct classification on the whole image
            # This is a fallback to the previous behavior
            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Get top prediction
                confidence, class_id = torch.max(probabilities, 0)
                confidence = confidence.item()
                class_id = class_id.item()
                
                top_prediction = {
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'Unknown',
                    'confidence': confidence,
                    'bbox': [0, 0, img.width, img.height]  # Full image
                }
                
                return {
                    'success': True,
                    'predictions': [top_prediction] if confidence >= CLASSIFICATION_THRESHOLD else [],
                    'has_prediction': confidence >= CLASSIFICATION_THRESHOLD,
                    'top_prediction': top_prediction,
                    'detection_mode': 'direct',
                    'num_detections': 0
                }
        
        # Stage 2: Classify each detected road sign
        results = []
        for detection in detections:
            bbox = detection['bbox']
            result = crop_and_classify(img, bbox)
            if result and result['has_prediction']:
                # Add detection confidence
                result['detection_confidence'] = detection['confidence']
                result['detection_class'] = detection['class']
                results.append(result)
        
        # Combine results
        all_predictions = []
        for result in results:
            for pred in result['predictions']:
                pred['bbox'] = result['bbox']  # Add bounding box to each prediction
                all_predictions.append(pred)
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'success': True,
            'predictions': all_predictions,
            'has_prediction': len(all_predictions) > 0,
            'top_prediction': all_predictions[0] if all_predictions else None,
            'detection_mode': 'two-stage',
            'num_detections': len(detections)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])  # Keep old route for backward compatibility
def predict():
    if request.method == 'POST':
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False, 
                'error': 'No image uploaded',
                'predictions': [],
                'has_prediction': False
            })
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                'success': False, 
                'error': 'No image selected',
                'predictions': [],
                'has_prediction': False
            })
        
        try:
            # Read image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            
            # Get predictions
            result = process_image(img)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': str(e),
                'predictions': [],
                'has_prediction': False
            })

@app.route('/api/predict_webcam', methods=['POST'])
@app.route('/predict_webcam', methods=['POST']) 
def predict_webcam():
    if request.method == 'POST':
        try:
            # Get the base64 encoded image from the request
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({
                    'success': False, 
                    'error': 'No image data received',
                    'predictions': [],
                    'has_prediction': False
                })
            
            # Decode the base64 image
            encoded_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            img_bytes = base64.b64decode(encoded_data)
            img = Image.open(io.BytesIO(img_bytes))
            
            # Get predictions
            result = process_image(img)
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'success': False, 
                'error': str(e),
                'predictions': [],
                'has_prediction': False
            })

# Simple route for Flutter health check
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': len(CLASS_NAMES)
    })

# Health check endpoint for Heroku
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': len(CLASS_NAMES) if CLASS_NAMES else 0
    })

# Add a simple index route if none of the template routes match
@app.route('/', methods=['GET'])
def index_redirect():
    try:
        return render_template('index.html')
    except Exception as e:
        return jsonify({
            'status': 'ok',
            'info': 'Road Sign Detection API is running. Use /api/predict or /realtime endpoints.',
            'error': str(e)
        })

if __name__ == '__main__':
    # Download models if needed
    print("Checking for model weights...")
    prepare_models()
    
    # Load the model
    print("Loading models...")
    load_model()
    
    # Get port from environment variable (Heroku sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)
