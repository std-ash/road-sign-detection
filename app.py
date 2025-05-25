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
ROAD_SIGN_CLASSES = ['traffic light', 'stop sign', 'parking meter', 'fire hydrant']

def create_model(num_classes):
    """Create a MobileNetV3 model for classification"""
    # Load a pre-trained MobileNetV3 Small model
    model = models.mobilenet_v3_small(weights=None)
    
    # Replace the classifier with a new one for our number of classes
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    return model

def load_yolo():
    """Load YOLOv5 model for object detection"""
    global yolo_model
    try:
        # Load YOLOv5 from torch hub
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        yolo_model.to(DEVICE)
        yolo_model.eval()
        print("YOLOv5 model loaded successfully")
        return yolo_model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        # Fallback to using YOLOv5 API if torch hub fails
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

def detect_road_signs(img):
    """Detect potential road signs in the image using YOLOv5"""
    try:
        if yolo_model is not None:
            # Use the loaded YOLOv5 model
            results = yolo_model(img)
            detections = results.pandas().xyxy[0]  # Get detection results as DataFrame
            
            # Filter detections for potential road signs
            road_sign_detections = []
            for _, detection in detections.iterrows():
                if detection['confidence'] >= DETECTION_THRESHOLD:
                    # Check if it's a known road sign class or has high confidence
                    if detection['name'] in ROAD_SIGN_CLASSES or detection['confidence'] > 0.7:
                        road_sign_detections.append({
                            'bbox': [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']],
                            'confidence': detection['confidence'],
                            'class': detection['name']
                        })
            
            return road_sign_detections
        else:
            # Fallback to using YOLOv5 API if model couldn't be loaded
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes = img_bytes.getvalue()
            
            # Use YOLOv5 API
            url = 'https://detect.roboflow.com/traffic-sign-detection-tsocd/1'
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
            }
            params = {
                'api_key': 'YOUR_API_KEY',  # Replace with your API key if using this method
                'confidence': DETECTION_THRESHOLD,
                'overlap': 0.5,
            }
            
            # This is a fallback and would require an API key to actually work
            # For now, we'll return an empty list to avoid errors
            return []
            
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
@app.route('/predict_webcam', methods=['POST'])  # Keep old route for backward compatibility
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

if __name__ == '__main__':
    load_model()
    # Get port from environment variable (Heroku sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
