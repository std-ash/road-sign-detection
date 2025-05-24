# Road Sign Detection API

A machine learning API for detecting and classifying road signs in images, designed for mobile integration.

## Features

- Two-stage detection approach:
  1. First detects road signs in images using YOLOv5
  2. Then classifies the detected signs using MobileNetV3
- Real-time detection via webcam
- REST API for mobile applications
- Flutter-ready with CORS support
- 75% confidence threshold for reliable predictions

## Project Structure

```
deployment/
├── app.py              # Main Flask application
├── model_loader.py     # Script to download model weights
├── classes.txt         # Road sign class names
├── requirements.txt    # Python dependencies
├── Procfile            # For Heroku deployment
├── runtime.txt         # Python version for Heroku
├── templates/          # Web interface templates
│   ├── index.html      # Image upload interface
│   └── realtime.html   # Real-time webcam detection
└── weights/            # Model weights directory
    └── best.pt         # Classification model (not in repo)
```

## Setup for Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your model weights in the `weights/` directory:
   - `weights/best.pt` - Your trained classification model
   - `weights/yolov5_traffic_signs.pt` - Optional specialized YOLOv5 model

3. Run the application:
   ```bash
   python app.py
   ```

4. Access the web interface at http://localhost:5000/

## API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/predict` - Submit an image for road sign detection
- `POST /api/predict_webcam` - Submit a base64-encoded image from webcam
- `GET /realtime` - Real-time webcam detection web interface

## Deployment to Heroku

### Prerequisites

1. [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. [Git](https://git-scm.com/)
3. Heroku account

### Steps

1. Initialize Git repository (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Create a Heroku app:
   ```bash
   heroku create your-app-name
   ```

3. Set up environment variables for your model URLs:
   ```bash
   heroku config:set MODEL_WEIGHTS_URL=https://your-storage-url/best.pt
   heroku config:set YOLO_WEIGHTS_URL=https://your-storage-url/yolov5_traffic_signs.pt
   ```

4. Deploy to Heroku:
   ```bash
   git push heroku main
   ```

5. Scale the web dyno:
   ```bash
   heroku ps:scale web=1
   ```

6. Open your app:
   ```bash
   heroku open
   ```

## Important Notes for Heroku Deployment

1. **Model Weights**: Due to GitHub and Heroku file size limitations, model weights are not included in the repository. The application will download them at startup using URLs specified in environment variables.

2. **Memory Usage**: YOLOv5 and PyTorch require significant memory. You might need to use a paid Heroku dyno for reliable performance.

3. **Slug Size**: Heroku has a 500MB slug size limit. We've optimized the dependencies, but you might need to further reduce them if you hit this limit.

4. **Timeout**: Heroku has a 30-second request timeout. The detection process should complete within this time, but for large images, it might time out.

## Flutter Integration

See the API documentation for details on integrating with Flutter applications. The API is designed to work seamlessly with mobile applications.

## License

[MIT](LICENSE)
