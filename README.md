# smart-assistant-girls-team

## CV Module - Face Detection

The CV module provides face detection functionality using OpenCV Haar Cascade classifier.

### Features Implemented
- Detect faces from input images.
- Return number of detected faces.
- Provide detection confidence score.
- Support multiple face detection.
- Handle invalid images and missing files.

### Stretch Features
Implemented:
- Rule-based image classification based on face detection results.
- Additional face detection evaluation using different models (YuNet and Res10 SSD).

Not Implemented:
- Advanced deep learning image classification.
- Real-time video face detection.

### Usage
The module can be accessed through the assistant API:

POST /assistant/image

Example request:
```json
{
    "image_path": "C:/smart-assistant-girls-team-1/tests/tests_images/one_face.jpg"
}
```

Example response:
```json
{
    "confidence": 0.91,
    "result": {
        "count": 1,
        "detected": true
    },
    "type": "face"
}
```