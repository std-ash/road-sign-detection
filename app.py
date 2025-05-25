import os
import io
import sys
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Basic dependencies that should always work
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import dependencies that might fail and handle gracefully
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms, models
    import cv2
    DEPS_LOADED = True
    logger.info("All ML dependencies loaded successfully")
except Exception as e:
    DEPS_LOADED = False
    logger.error(f"Error loading ML dependencies: {e}")

# Import optional dependencies
try:
    import requests
    import time
except Exception as e:
    logger.warning(f"Optional dependencies not loaded: {e}")

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for Flutter integration

# Initialize global variables
model = None
yolo_model = None
CLASS_NAMES = []
DEPS_LOADED_SUCCESSFULLY = False

# Image preprocessing transformer (only define if dependencies loaded)
if DEPS_LOADED:
    # Load the trained model
    MODEL_PATH = os.path.join('weights', 'best.pt')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
else:
    logger.warning("ML dependencies not loaded. App will run in limited mode.")

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
    if not DEPS_LOADED:
        logger.error("Cannot load YOLOv5 - dependencies not available")
        return None
        
    try:
        # Load YOLOv5n (nano) - the smallest and fastest model
        logger.info("Loading YOLOv5n model...")
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, force_reload=False)
        # Optimize for CPU inference
        yolo_model.to(DEVICE)
        yolo_model.eval()
        # Set lower inference size to reduce memory usage
        yolo_model.conf = 0.45  # Higher confidence threshold
        yolo_model.iou = 0.45   # Higher IoU threshold
        logger.info("YOLOv5n model loaded successfully")
        return yolo_model
    except Exception as e:
        logger.error(f"Error loading YOLOv5 model: {e}")
        return None

def load_model():
    """Load models with proper error handling"""
    global model, yolo_model, CLASS_NAMES, DEPS_LOADED_SUCCESSFULLY
    
    if not DEPS_LOADED:
        logger.error("Cannot load models - dependencies not available")
        return None
    
    try:
        # Load class names
        try:
            with open('classes.txt', 'r') as f:
                CLASS_NAMES = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(CLASS_NAMES)} class names")
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            # Create some dummy class names to prevent crashes
            CLASS_NAMES = [f"Class_{i}" for i in range(10)]
        
        # Load classification model
        try:
            num_classes = len(CLASS_NAMES)
            logger.info(f"Creating model for {num_classes} classes...")
            model = create_model(num_classes)
            
            if os.path.exists(MODEL_PATH):
                logger.info(f"Loading model weights from {MODEL_PATH}...")
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
                logger.info("Classification model loaded successfully")
            else:
                logger.warning(f"Warning: Model not found at {MODEL_PATH}. Using untrained model.")
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            model = None
        
        # Try to load YOLOv5 model but don't let it crash the app
        try:
            # Load YOLOv5 detection model with a timeout
            logger.info("Attempting to load YOLOv5 model...")
            yolo_model = load_yolo()
        except Exception as e:
            logger.error(f"Error in YOLOv5 loading process: {e}")
            yolo_model = None
        
        # Mark as successful if at least one model loaded
        if model is not None or yolo_model is not None:
            DEPS_LOADED_SUCCESSFULLY = True
            logger.info("At least one model loaded successfully")
            return model
        else:
            logger.error("Failed to load any models")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in model loading: {e}")
        return None

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

@app.route('/status')
def status():
    """Provide app status information without relying on ML models"""
    status_info = {
        'app_running': True,
        'dependencies_loaded': DEPS_LOADED,
        'models_loaded': DEPS_LOADED_SUCCESSFULLY,
        'classification_model': model is not None,
        'detection_model': yolo_model is not None,
        'num_classes': len(CLASS_NAMES) if CLASS_NAMES else 0,
        'server_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version,
    }
    return jsonify(status_info)

@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])  # Keep old route for backward compatibility
def predict():
    if request.method == 'POST':
        # Check if ML dependencies and models are loaded
        if not DEPS_LOADED:
            logger.error("ML dependencies not loaded, cannot make predictions")
            return jsonify({
                'success': False,
                'error': 'ML dependencies not available',
                'status': 'limited_mode'
            })
        
        if not DEPS_LOADED_SUCCESSFULLY:
            logger.error("Models failed to load, cannot make predictions")
            return jsonify({
                'success': False,
                'error': 'Models failed to load properly',
                'status': 'model_error'
            })
        
        try:
            # Get image from request
            if 'file' not in request.files:
                logger.warning("No file found in request")
                return jsonify({
                    'success': False,
                    'error': 'No file found'
                })
                
            file = request.files['file']
            if file.filename == '':
                logger.warning("Empty filename received")
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                })
                
            # Read and process the image
            try:
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes))
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                logger.info(f"Image processed successfully: {img.size}")
            except Exception as img_error:
                logger.error(f"Error processing image: {img_error}")
                return jsonify({
                    'success': False,
                    'error': f'Error processing image: {str(img_error)}'
                })
            
            # Process the image using YOLO detection followed by classification
            try:
                predictions = process_image(img_cv)
                logger.info(f"Processed image with {len(predictions) if predictions else 0} predictions")
            except Exception as pred_error:
                logger.error(f"Error during prediction: {pred_error}")
                return jsonify({
                    'success': False,
                    'error': f'Error during prediction: {str(pred_error)}'
                })
            
            # Return predictions in the desired format
            if predictions and len(predictions) > 0:
                # Sort by confidence (descending)
                predictions.sort(key=lambda x: x['confidence'], reverse=True)
                # Find predictions with confidence above threshold
                valid_predictions = [p for p in predictions if p['confidence'] >= CLASSIFICATION_THRESHOLD]
                
                response = {
                    'success': True,
                    'predictions': predictions,
                    'has_prediction': len(valid_predictions) > 0,
                    'top_prediction': valid_predictions[0] if valid_predictions else None,
                }
            else:
                response = {
                    'success': True,
                    'predictions': [],
                    'has_prediction': False,
                    'top_prediction': None,
                }
            
            logger.info(f"Returning prediction response with success={response['success']}")
            return jsonify(response)
        except Exception as e:
            logger.error(f"Unexpected error in prediction endpoint: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            })

@app.route('/api/predict_webcam', methods=['POST'])
@app.route('/predict_webcam', methods=['POST'])  # Keep old route for backward compatibility
def predict_webcam():
    if request.method == 'POST':
        # Check if ML dependencies and models are loaded
        if not DEPS_LOADED:
            logger.error("ML dependencies not loaded, cannot make predictions")
            return jsonify({
                'success': False,
                'error': 'ML dependencies not available',
                'status': 'limited_mode'
            })
        
        if not DEPS_LOADED_SUCCESSFULLY:
            logger.error("Models failed to load, cannot make predictions")
            return jsonify({
                'success': False,
                'error': 'Models failed to load properly',
                'status': 'model_error'
            })
        
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

# Fix the mismatch in file parameter names
@app.before_request
def fix_image_param():
    if request.method == 'POST' and request.path in ['/predict', '/api/predict']:
        if 'image' in request.files and 'file' not in request.files:
            # Copy 'image' param to 'file' param for compatibility
            request.files = dict(request.files)
            request.files['file'] = request.files['image']

# Initialize the app with models if possible
def initialize_app():
    """Initialize the application with careful error handling"""
    try:
        if DEPS_LOADED:
            logger.info("Attempting to load models...")
            load_model()
        else:
            logger.warning("Skipping model loading due to missing dependencies")
    except Exception as e:
        logger.error(f"Error during app initialization: {e}")
    
    logger.info("Application initialized and ready to serve requests")

# Initialize on startup
initialize_app()

if __name__ == '__main__':
    # Use the PORT environment variable if available (for Heroku deployment)
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
