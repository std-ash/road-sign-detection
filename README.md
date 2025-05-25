# Road Sign Detection API

A real-time road sign detection system with two-stage detection: YOLOv5 for locating signs and MobileNetV3 for classification.

## Features

- Two-stage detection approach:
  - YOLOv5 for detecting road signs from a distance
  - MobileNetV3 for accurately classifying the detected signs
- Real-time webcam detection via browser
- REST API for mobile integration (Flutter)
- 75% confidence threshold for reliable predictions

## Live Demo

[Access the demo here](#) (Update this link after Heroku deployment)

## API Endpoints

### Health Check
```
GET /api/health
```

### Image Upload Detection
```
POST /api/predict
```

### Real-time Webcam Detection
```
POST /api/predict_webcam
```

## Local Development

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python app.py
```

## Deployment

This application is ready for deployment on Heroku:

1. Create a new Heroku app
2. Connect your GitHub repository
3. Enable the Heroku Buildpacks:
   - `heroku/python`
   - `https://github.com/heroku/heroku-buildpack-apt`
4. Deploy the application

## Flutter Integration

Check the API response format for integration with mobile applications:

```json
{
  "success": true,
  "predictions": [
    {
      "class_id": 5,
      "class_name": "Stop Sign",
      "confidence": 0.97,
      "bbox": [100, 200, 300, 400]
    }
  ],
  "has_prediction": true,
  "top_prediction": {
    "class_id": 5,
    "class_name": "Stop Sign",
    "confidence": 0.97,
    "bbox": [100, 200, 300, 400]
  },
  "detection_mode": "two-stage",
  "num_detections": 1
}
```

## License

MIT
