import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
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
    return jsonify({
        'status': 'ok',
        'info': 'Road Sign Detection API is running. Use /api/predict or /api/health endpoints.'
    })

# Realtime page
@app.route('/realtime')
def realtime():
    return jsonify({
        'status': 'ok',
        'info': 'Realtime endpoint is available in the full version.'
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

# Placeholder webcam endpoint
@app.route('/api/predict_webcam', methods=['POST'])
@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
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

if __name__ == '__main__':
    # Get port from environment variable (Heroku sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)
