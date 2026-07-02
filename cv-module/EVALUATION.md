Day (1): 
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

Day(2):ScaleFactor Evaluation
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

Day(3):Parameter Combination Evaluation
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

Day(4):Eye Detection Validation

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