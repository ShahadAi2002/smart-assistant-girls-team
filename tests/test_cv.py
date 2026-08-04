import pytest
import sys
import os

sys.path.append(os.path.abspath("cv-module"))

from face_detector import detect_faces
from image_classifier import classify_image


#Face detection
def test_no_face():

    result = detect_faces("tests/tests_images/no_face.jpg")

    assert result["result"]["count"] == 0
    assert result["result"]["detected"] == False


def test_one_face():

    #Original 
    result = detect_faces("tests/tests_images/one_face.jpg")
    assert result["result"]["count"] == 1
    assert result["result"]["detected"] == True

    #Face with glasses
    result = detect_faces("tests/tests_images/glasses.jpg")
    assert result["result"]["count"] == 1
    assert result["result"]["detected"] == True


def test_two_faces():
    result = detect_faces("tests/tests_images/two_faces.jpg")

    assert result["result"]["count"] == 2
    assert result["result"]["detected"] == True


#Error Handling tests

def test_missing_file():
    #Missing
    result = detect_faces("tests/tests_images/missing.jpg")

    assert result["result"]["count"] == 0
    assert result["result"]["detected"] is False
    assert result["confidence"] == 0.0


def test_corrupted_image():
    
    #Corrupted image
    result = detect_faces("tests/tests_images/corrupted file.jpg")

    assert result["result"]["count"] == 0
    assert result["result"]["detected"] == False
    assert result["confidence"] == 0.0


def test_empty_image():
    #Empty image
    result = detect_faces("tests/tests_images/empty.jpg")

    assert result["result"]["count"] == 0
    assert result["result"]["detected"] == False
    assert result["confidence"] == 0.0

def test_fake_image():
    #File not image
    result = detect_faces("tests/tests_images/fake.jpg")

    assert result["result"]["count"] == 0
    assert result["result"]["detected"] == False
    assert result["confidence"] == 0.0


#Image Classification
def test_contains_faces():

    result = classify_image("tests/tests_images/one_face.jpg")

    assert result["result"]["label"]=="Contains Faces"


def test_no_faces():

    result = classify_image("tests/tests_images/no_face.jpg")

    assert result["result"]["label"]=="No Faces"   