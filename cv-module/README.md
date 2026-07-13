#CV Module

## Overview
he CV module provides basic computer vision functionality for the Smart Assistant project using OpenCV. It supports face detection, drawing bounding boxes around detected faces, and validating face detection by detecting eyes within each detected face.

## Requirements
The following packages are required to run the CV module:

- Python 3.x
- OpenCV (`opencv-python`)

## File Structure

```
cv-module/
├── face_detector.py
├── image_classifier.py
└── README.md
```
- **face_detector.py**: Implements face detection, face drawing, and eye detection functions.
- **image_classifier.py**: Contains a basic rule-based image classifier (Stretch Goal).
- **README.md**: Documentation for the CV module.

## Functions
1- detect_faces(image_path)

Detects whether the input image contains faces using the OpenCV Haar Cascade classifier.

**Input:**
- `image_path` (str): Path to the input image.

**Output:**
```json
{
  "type": "face",
  "result": {
    "detected": true,
    "count": 1
  },
  "confidence": 0.91
}
```
2- draw_faces(image_path, output_path)

Draws bounding boxes around the detected faces in the input image and saves the processed image.

**Input:**
- `image_path` (str): Path to the input image.
- `output_path` (str): Path where the output image will be saved.

**Output:**
- `None`

**Description:**
- Detects faces in the image.
- Draws a bounding box around each detected face.
- Saves the processed image to the specified output path.
3- detect_faces_with_eyes(image_path)

Detects faces and counts the number of eyes detected within each detected face using the Haar Cascade eye classifier.

**Input:**
- `image_path` (str): Path to the input image.

**Output:**

- `list[int]`

Example:

```python
[2, 2, 1]
```

**Description:**
- Detects all faces in the image.
- Detects eyes inside each detected face.
- Returns the number of detected eyes for each face.

## How to run
1. Place the input image in your project directory.
2. Import the required function from `face_detector.py`.
3. Call the desired function with the image path.

## Example

### Detect faces
```python
from face_detector import detect_faces

result = detect_faces("images/person.jpg")
print(result)
```

### Draw faces
```python
from face_detector import draw_faces

draw_faces("images/person.jpg", "output/result.jpg")
```

### Detect faces with eyes
```python
from face_detector import detect_faces_with_eyes

eyes = detect_faces_with_eyes("images/person.jpg")
print(eyes)
```
## Current limitation
- Works best with clear front-facing faces.
- Detection accuracy decreases in low-light conditions.
- Performance may decrease for side-profile faces.
- Uses Haar Cascade, which is less robust than modern deep learning models.