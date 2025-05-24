import os
import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)  # Enable CORS for Flutter integration

# Basic class names for demonstration
CLASS_NAMES = [
    'Bus-stop', 'Compulsory-Roundabout', 'Cross-Roads-Ahead',
    'Double-Bend-to-Left-Ahead', 'Double-Bend-to-Right-Ahead',
    'Falling-Rocks-Ahead', 'Left-Bend-Ahead', 'Level-crossing-with-barriers-ahead',
    'Level-crossing-without-barriers-ahead', 'Narrow-Bridge-or-Culvert-Ahead',
    'No-entry', 'No-overtaking', 'No-parking', 'Pedestrian-crossing-ahead',
    'Right-Bend-Ahead', 'Road-narrows-on-both-sides-ahead', 'School-ahead',
    'Slippery-Road-Ahead', 'Speed-limit-20', 'Speed-limit-30', 'Speed-limit-50',
    'Stop', 'T-Junction-Ahead', 'Traffic-signals-ahead'
]

# Home page
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return jsonify({
            'status': 'ok',
            'info': 'Road Sign Detection API is running. Use /api/predict or /realtime endpoints.',
            'error': str(e)
        })

# Realtime page
@app.route('/realtime')
def realtime():
    try:
        return render_template('realtime.html')
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Health check endpoint
@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Road Sign Detection API is running in lightweight mode',
        'classes': len(CLASS_NAMES)
    })

# Placeholder prediction endpoint
@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Check if an image was uploaded
            if 'image' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No image uploaded',
                    'predictions': [],
                    'has_prediction': False
                })
            
            # Get the image from the request
            file = request.files['image']
            img = Image.open(file.stream)
            
            # Return a placeholder response
            return jsonify({
                'success': True,
                'message': 'This is a lightweight placeholder. ML models are not loaded in this version.',
                'predictions': [
                    {
                        'class': 'Placeholder Detection',
                        'confidence': 0.99,
                        'bbox': [10, 10, 100, 100]
                    }
                ],
                'has_prediction': True,
                'top_prediction': {
                    'class': 'Placeholder Detection',
                    'confidence': 0.99
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'predictions': [],
                'has_prediction': False
            })

# Placeholder webcam endpoint
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
            
            # Return a placeholder response
            return jsonify({
                'success': True,
                'message': 'This is a lightweight placeholder. ML models are not loaded in this version.',
                'predictions': [
                    {
                        'class': 'Placeholder Detection',
                        'confidence': 0.99,
                        'bbox': [10, 10, 100, 100]
                    }
                ],
                'has_prediction': True,
                'top_prediction': {
                    'class': 'Placeholder Detection',
                    'confidence': 0.99
                }
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'predictions': [],
                'has_prediction': False
            })

if __name__ == '__main__':
    # Get port from environment variable (Heroku sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)
