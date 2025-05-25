import os
import io
import base64
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from torchvision import transforms, models

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for Flutter integration

# Load the trained model
MODEL_PATH = os.path.join('weights', 'best.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
CLASS_NAMES = []

# Image preprocessing transformer
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Confidence threshold for showing predictions (75%)
CONFIDENCE_THRESHOLD = 0.75

def create_model(num_classes):
    """Create a MobileNetV3 model for classification"""
    # Load a pre-trained MobileNetV3 Small model
    model = models.mobilenet_v3_small(weights=None)
    
    # Replace the classifier with a new one for our number of classes
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    return model

def load_model():
    global model, CLASS_NAMES
    # Load class names
    with open('classes.txt', 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    
    # Load model
    num_classes = len(CLASS_NAMES)
    model = create_model(num_classes)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Using untrained model.")
    
    return model

def process_image(img):
    """Process an image and return predictions only if confidence exceeds threshold"""
    try:
        # Ensure the image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transforms
        img_tensor = transform(img)
        
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
                if confidence >= CONFIDENCE_THRESHOLD or i == 0:  # Always include top prediction
                    predictions.append({
                        'class_id': class_id,
                        'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'Unknown',
                        'confidence': confidence
                    })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Filter out predictions below threshold
            filtered_predictions = [p for p in predictions if p['confidence'] >= CONFIDENCE_THRESHOLD]
            
            return {
                'success': True,
                'predictions': filtered_predictions,
                'has_prediction': len(filtered_predictions) > 0,
                'top_prediction': predictions[0] if predictions else None
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
    # Get port from environment variable (Heroku sets this) or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
