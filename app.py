import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

# Use TFLite runtime instead of full TensorFlow
# We'll need a dummy interpreter class for both missing model files and missing TF/TFLite
class DummyInterpreter:
    def __init__(self, model_path=None):
        self.model_path = model_path
        print(f"Dummy interpreter created with model path: {model_path}")
        
    def allocate_tensors(self):
        print("Dummy allocate_tensors called")
        
    def get_input_details(self):
        return [{'shape': [1, 224, 224, 3], 'index': 0}]
        
    def get_output_details(self):
        return [{'shape': [1, 37], 'index': 0}]
        
    def set_tensor(self, index, tensor):
        print(f"Dummy set_tensor called with index {index}")
        
    def invoke(self):
        print("Dummy invoke called")
        
    def get_tensor(self, index):
        print(f"Dummy get_tensor called with index {index}")
        # Return random predictions for testing
        return np.random.random((1, 37))

# Try to import the TFLite interpreter
try:
    # Import TFLite interpreter from tflite_runtime package
    from tflite_runtime.interpreter import Interpreter
    print("TFLite runtime imported successfully")
    HAS_TFLITE = True
except ImportError:
    # Fall back to TensorFlow if tflite_runtime is not available
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        print("TensorFlow Lite imported from full TensorFlow package")
        HAS_TFLITE = True
    except ImportError:
        print("WARNING: Neither TFLite runtime nor TensorFlow is available")
        # Use our dummy interpreter
        Interpreter = DummyInterpreter
        print("Using dummy interpreter for testing")
        HAS_TFLITE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter integration

# Constants
MODEL_PATH = os.path.join('model.tflite')
CLASS_PATH = os.path.join('classes.txt')
CONFIDENCE_THRESHOLD = 0.75  # 75% threshold for predictions

# Global variables
interpreter = None
class_names = []
input_details = None
output_details = None

def load_model():
    """Load the TFLite model and class names"""
    global interpreter, class_names, input_details, output_details
    
    try:
        print(f"Current working directory: {os.getcwd()}")
        print(f"Listing files in current directory: {os.listdir('.')}")
        
        # Load class names
        if os.path.exists(CLASS_PATH):
            print(f"Found classes file at {CLASS_PATH}")
            with open(CLASS_PATH, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(class_names)} classes")
        else:
            print(f"Warning: Classes file not found at {CLASS_PATH}")
            # Fallback to default classes
            class_names = [f"Class_{i}" for i in range(37)]
            print(f"Using {len(class_names)} fallback classes")
        
        # Load TFLite model
        if os.path.exists(MODEL_PATH) and HAS_TFLITE:
            print(f"Loading model from {MODEL_PATH}")
            interpreter = Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            
            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"Model loaded successfully with:")
            print(f"- Input shape: {input_details[0]['shape']}")
            print(f"- Output shape: {output_details[0]['shape']}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH} or TFLite not available. Using dummy model.")
            interpreter = DummyInterpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Create a dummy interpreter for testing
        interpreter = DummyInterpreter(model_path="dummy_model")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        # Get input shape
        input_shape = input_details[0]['shape']
        
        # Model expects: [1, height, width, channels]
        required_height, required_width = input_shape[1], input_shape[2]
        
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to match model input
        image = image.resize((required_width, required_height))
        
        # Convert PIL Image to numpy array and normalize to 0-1
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        # Return dummy input of correct shape
        return np.zeros(input_details[0]['shape'], dtype=np.float32)

def process_image(img):
    """Process an image and return predictions"""
    try:
        # Check if model is loaded
        global interpreter, class_names, input_details, output_details
        if interpreter is None:
            print("Warning: Model not loaded, attempting to load it")
            load_model()
            if interpreter is None:
                return {
                    'success': False,
                    'error': 'Model not loaded',
                    'predictions': [],
                    'has_prediction': False
                }
        
        # Preprocess the image
        processed_image = preprocess_image(img)
        
        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Run the inference
        interpreter.invoke()
        
        # Get the output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Convert outputs to probabilities with softmax
        probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        
        # Get top predictions
        # Ensure we don't try to get more classes than we have
        k = min(3, len(class_names))
        indices = np.argsort(probabilities[0])[-k:][::-1]  # Top k indices in descending order
        probs = probabilities[0][indices]  # Corresponding probabilities
        
        predictions = []
        for i, (idx, prob) in enumerate(zip(indices, probs)):
            # Only include predictions with confidence above threshold or the top one
            if prob >= CONFIDENCE_THRESHOLD or i == 0:  # Always include top prediction
                predictions.append({
                    'class_id': int(idx),
                    'class_name': class_names[idx] if idx < len(class_names) else f"Class_{idx}",
                    'confidence': float(prob)
                })
        
        # Filter out predictions below threshold
        filtered_predictions = [p for p in predictions if p['confidence'] >= CONFIDENCE_THRESHOLD]
        
        return {
            'success': True,
            'predictions': filtered_predictions,
            'has_prediction': len(filtered_predictions) > 0,
            'top_prediction': predictions[0] if predictions else None
        }
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'predictions': [],
            'has_prediction': False
        }

@app.route('/api/predict', methods=['POST'])
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

# Simple route for Flutter health check and documentation
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': interpreter is not None,
        'classes': len(class_names),
        'api_endpoints': [
            {
                'path': '/api/predict',
                'method': 'POST',
                'description': 'Upload an image file for road sign detection',
                'parameter': 'image (form-data file)'  
            },
            {
                'path': '/api/predict_webcam',
                'method': 'POST',
                'description': 'Send base64-encoded image for road sign detection',
                'parameter': 'image (JSON string)'  
            },
            {
                'path': '/api/health',
                'method': 'GET',
                'description': 'Check API status and available endpoints'  
            }
        ]
    })

# Root route that documents the API
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'name': 'Road Sign Detection API',
        'description': 'API for detecting road signs in images',
        'version': '1.0.0',
        'endpoints': [
            '/api/predict - POST: Detect road signs in uploaded image',
            '/api/predict_webcam - POST: Detect road signs in base64 image',
            '/api/health - GET: Check API status'
        ],
        'status': 'online'
    })

# Initialize model at startup (but not during module import)
interpreter = None
class_names = []

# This will ensure the model is loaded when the app starts, not just in __main__
@app.before_first_request
def initialize():
    global interpreter
    if interpreter is None:
        print("Loading model before first request")
        load_model()

if __name__ == '__main__':
    # Get port from environment variable (Heroku sets this) or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # In development, load the model at startup
    load_model()
    app.run(debug=False, host='0.0.0.0', port=port)
