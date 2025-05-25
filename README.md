# Road Sign Detection API

A real-time road sign detection API built with Flask and PyTorch, ready for Flutter mobile integration.

## Features

- Real-time road sign detection using a trained MobileNetV3 model
- REST API with 75% confidence threshold for reliable predictions
- Web interface for testing with image upload and webcam
- Mobile-ready endpoints for Flutter integration

## API Endpoints

- `GET /api/health` - Check API status
- `POST /api/predict` - Upload an image for detection
- `POST /api/predict_webcam` - Send base64-encoded image for detection

## Flutter Integration

Example code for Flutter integration:

```dart
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

Future<void> detectRoadSign() async {
  final picker = ImagePicker();
  final image = await picker.pickImage(source: ImageSource.camera);
  
  if (image == null) return;
  
  var request = http.MultipartRequest('POST', Uri.parse('https://your-api-url.herokuapp.com/api/predict'));
  request.files.add(await http.MultipartFile.fromPath('image', image.path));
  
  var response = await request.send();
  if (response.statusCode == 200) {
    var responseData = await response.stream.bytesToString();
    // Parse JSON and handle the prediction result
    print(responseData);
  }
}
```

## Deployment

### Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   python app.py
   ```

### Heroku Deployment

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

## Response Format

```json
{
  "success": true,
  "predictions": [
    {
      "class_id": 5,
      "class_name": "Stop Sign",
      "confidence": 0.92
    }
  ],
  "has_prediction": true,
  "top_prediction": {
    "class_id": 5,
    "class_name": "Stop Sign",
    "confidence": 0.92
  }
}
```

## License

MIT
