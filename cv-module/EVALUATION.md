# CV Module Evaluation

## 1. Face Detection Evaluation (Haar Cascade)
-The initial implementation using Haar Cascade produced good results on normal frontal faces. -several false positives were observed in images without faces and in challenging lighting conditions.
-Face detection accuracy also decreased when faces were rotated.
Observation
| Image               | Scale Factor | Min Neighbors | Min Size | Face Detect | Correct | Notes      |
|---------------------|--------------|---------------|----------|-------------|---------|------------|
| no_face             | 1.1          | 5             | 30,30    | 1           | FP (0)  |            |
| one_face            | 1.1          | 5             | 30,30    | 1           | TRUE (1)|            |
| Two_faces           | 1.1          | 5             | 30,30    | 2           | TRUE (2)|            |
| group_faces         | 1.1          | 5             | 30,30    | 8           | TRUE (8)|            |
| Changing_lighting   | 1.1          | 5             | 30,30    | 3           | FP (1)  |            |
| low_light           | 1.1          | 5             | 30,30    | 1           | TRUE (1)|            |
| diff_angle          | 1.1          | 5             | 30,30    | 9           | FP (9)  | Can't detect 90° face |

# ScaleFactor Evaluation
Several scaleFactor values were tested to evaluate their effect on detection accuracy.
| Image               | Actual | 1.02 | 1.05 | 1.1 | 1.2 | 1.3 |
|---------------------|:------:|:----:|:----:|:---:|:---:|:---:|
| no_face             |   0    |  1   |  1   |  1  |  0  |  0  |
| one_face            |   1    |  1   |  1   |  1  |  1  |  1  |
| Two_faces           |   2    |  3   |  2   |  2  |  2  |  2  |
| group_faces         |   8    | 10   |  9   |  8  |  8  |  8  |
| Changing_lighting   |   1    |  4   |  3   |  3  |  1  |  0  |
| low_light           |   1    |  1   |  1   |  1  |  1  |  1  |
| diff_angle          |  15    | 17   | 11   |  9  |  8  |  7  |

**Observation:**
- Smaller `scaleFactor` values (1.02 and 1.05) detected more faces but also produced more false positives.
- Increasing the `scaleFactor` generally reduced false-positive detections.
- `scaleFactor = 1.2` provided the best balance for the tested images by reducing false positives while maintaining good detection accuracy.

# Parameter Combination Evaluation
After evaluating minNeighbors and minSize individually, different parameter combinations were tested to identify the most suitable configuration for face detection.

1:miniNeighbors
| Image             | Actual | 3 | 5 | 7 | 9 |
|-------------------|:------:|:-:|:-:|:-:|:-:|
| no_face           | 0 | 0 | 0 | 0 | 0 |
| one_face          | 1 | 1 | 1 | 1 | 1 |
| Two_faces         | 2 | 2 | 2 | 2 | 2 |
| group_faces       | 8 | 9 | 8 | 8 | 8 |
| Changing_lighting | 1 | 2 | 1 | 0 | 0 |
| low_light         | 1 | 1 | 1 | 1 | 1 |
| diff_angle        | 15 | 9 | 8 | 7 | 7 |

2:miniSize
| Image             | Actual | 20,20 | 30,30 | 40,40 | 50,50 |
|-------------------|:------:|:-----:|:-----:|:-----:|:-----:|
| no_face           | 0 | 0 | 0 | 0 | 0 |
| one_face          | 1 | 1 | 1 | 1 | 1 |
| Two_faces         | 2 | 2 | 2 | 2 | 2 |
| group_faces       | 8 | 8 | 8 | 8 | 8 |
| Changing_lighting | 1 | 1 | 1 | 1 | 1 |
| low_light         | 1 | 1 | 1 | 1 | 1 |
| diff_angle        | 15 | 8 | 8 | 8 | 8 |

3:Parameter combinations
| Parameters        | no_face | one_face | Two_faces | group_faces | Changing_lighting | low_light | diff_angle |
|-------------------|:-------:|:--------:|:---------:|:-----------:|:-----------------:|:---------:|:----------:|
| 1.1, 5, 30×30     | 1 | 1 | 2 | 8 | 3 | 1 | 9 |
| 1.1, 7, 30×30     | 0 | 1 | 2 | 8 | 1 | 1 | 9 |
| 1.1, 5, 40×40     | 1 | 1 | 2 | 8 | 3 | 1 | 9 |
| 1.1, 7, 40×40     | 0 | 1 | 2 | 8 | 1 | 1 | 9 |
| 1.2, 5, 30×30     | 0 | 1 | 2 | 8 | 1 | 1 | 8 |
| 1.2, 7, 30×30     | 0 | 1 | 2 | 8 | 0 | 1 | 7 |
| 1.2, 5, 40×40     | 0 | 1 | 2 | 8 | 1 | 1 | 8 |
| 1.2, 7, 40×40     | 0 | 1 | 2 | 8 | 0 | 1 | 7 |

Observations
Increasing minNeighbors reduced false positive detections.
Changing minSize had only a minor effect on the tested images.
Testing parameter combinations produced better results than modifying a single parameter independently.
The combination below achieved the best overall performance:
scaleFactor = 1.2
minNeighbors = 7
minSize = (30,30)

This combination minimized false positives while maintaining good face detection accuracy across most test images.

# Eye Detection Validation

Eye detection was added using haarcascade_eye.xml to validate detected faces.
The best eye detection parameters were:
scaleFactor = 1.02
minNeighbors = 3
minSize = (15,15)

Observations:
-Eye detection worked well for frontal faces.
-Detection accuracy decreased under low-light conditions and different face angles.
-Detecting eyes helped verify detected faces and provided an additional validation step.
-Some faces were detected successfully, but only one eye was found due to lighting or pose variations.

Finally, Parameter tuning significantly improved the face detection performance.Using eye detection as a secondary validation step increased confidence in detected faces, although challenging lighting conditions and face rotations still affected the overall accuracy.


## 2. Rule-Based Image Classification Evaluation

### Description

A simple rule-based image classifier was implemented using the output of `detect_faces()`. Images are classified into two categories:
- Contains Faces
- No Faces

The classifier depends entirely on the face detection results returned by the Haar Cascade-based face detector.

**Note**: The confidence score follows the project's interface specification. A fixed value (0.91) is returned when one or more faces are detected, while a value of (0.0) is returned when no faces are detected. Therefore, it should not be interpreted as a probabilistic confidence score.

### Test Results

| Test Image | Expected Face Count | Actual Face Count | Confidence | Correct? |
|------------|--------------------|-------------------|------------|----------|
| no_face.jpg | 0 | 0 | 0.00 | Yes |
| one_face.jpg | 1 | 1 | 0.91 | Yes |
| two_faces.jpg | 2 | 2 | 0.91 | Yes |
| group.jpg | 8 | 8 | 0.91 | Yes |
| low.jpg | 1 | 1 | 0.91 | Yes |
| lowlight.jpg | 1 | 1 | 0.91 | Yes |
| maxresdefault.jpg | 15 | 9 | 0.91 | No |

### Accuracy

- Correct classifications: 6 / 7
- Classification accuracy: 85.7% 
### Strengths

- Produces consistent results on clear images. 
- Successfully detects single and multiple frontal faces.
- Fast and simple rule-based implementation.

### Limitations

- Limited to two classes only (Contains Faces / No Faces).
- Performance depends entirely on the underlying Haar Cascade detector.
- Missed several faces in images containing many faces or challenging viewing angles. 
- The confidence score is fixed by the project interface specification and does not represent a probabilistic confidence estimate.

## 3. OpenCV DNN Evaluation (Res10 SSD)
### Description

A face detection model based on OpenCV's DNN module and the pre-trained Res10 SSD Caffe model was implemented. The model detects faces in images and returns the number of detected faces, bounding boxes, and an average confidence score for the detections.

### Test Results

| Test Image | Expected Face Count | Actual Face Count | Average Confidence | Correct? |
|------------|--------------------|-------------------|--------------------|----------|
| no_face.jpg | 0 | 0 | 0.00 | Yes |
| one_face.jpg | 1 | 1 | 0.90 | Yes |
| two_faces.jpg | 2 | 2 | 1.00 | Yes |
| group.jpg | 8 | 3 | 0.65 | No |
| low.jpg | 1 | 1 | 1.00 | Yes |
| lowlight.jpg | 1 | 1 | 1.00 | Yes |
| maxresdefault.jpg | 15 | 8 | 0.77 | No |

### Accuracy

- Correct face counts: 5 / 7
- Face count accuracy: 71.4% on the tested images.
- Successfully detected at least one face in all images containing faces.

### Execution Time

- Average execution time ranged approximately from 0.13 to 0.18 seconds on the tested images.

### Strengths

- Provides probabilistic confidence scores for detections.
- Performs well on images containing one or two faces.
- Successfully detects faces under low-light conditions.
- Fast execution time across all tested images.
- Produces bounding boxes for detected faces.

### Limitations

- Missed several faces in crowded or group images.
- Face count accuracy decreases as the number of faces increases.
- Detection performance varies depending on image complexity.


## 4. YuNet Face Detection Evaluation
### Description
A face detection model based on OpenCV's YuNet deep learning model was implemented. The model detects faces in images and returns the number of detected faces, bounding boxes, and an average confidence score for the detections.

### Test Results

| Test Image | Expected Face Count | Actual Face Count | Average Confidence | Correct? |
|------------|--------------------|-------------------|--------------------|----------|
| no_face.jpg | 0 | 0 | 0.00 | Yes |
| one_face.jpg | 1 | 1 | 0.95 | Yes |
| two_faces.jpg | 2 | 2 | 0.94 | Yes |
| group.jpg | 8 | 8 | 0.94 | Yes |
| low.jpg | 1 | 1 | 0.93 | Yes |
| lowlight.jpg | 1 | 0 | 0.00 | No |
| maxresdefault.jpg | 15 | 14 | 0.92 | No |


### Accuracy
- Correct face counts: 5 / 7 
- Face count accuracy: 71.4%

### Execution Time

- Average execution time was approximately 0.0318 seconds on the tested images.

### Strengths

- Achieved excellent performance on images containing multiple faces. 
- Successfully detected 14 out of 15 faces in an image with significant pose variations.
- Provides confidence scores and bounding boxes for detected faces.

### Limitations

- Failed to detect the face in the low-light image.
 -  May miss faces in challenging lighting conditions.
- Face count accuracy may vary depending on image quality and pose variations.


### Overall Comparison

| Model | Correct Results | Accuracy |
|-------|-----------------|----------|
| Rule-Based | 6 / 7 | 85.7% |
| Res10 SSD | 5 / 7 | 71.4% |
| YuNet | 5 / 7 | 71.4% |

