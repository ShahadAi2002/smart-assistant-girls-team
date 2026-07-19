from face_detector import detect_faces

def classify_image (image_path):
   """
    Classify an image based on face detection results.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: A dictionary containing the image classification result.
    """
   face_detection_result = detect_faces(image_path)
   face_count = face_detection_result["result"]["count"]
   confidence = face_detection_result["confidence"]

   #base rule
   if face_count > 0:
      label = "Contains Faces"

   else:
      label = "No Faces"   

   return  {"type": "image_classification",
            "result": {"label": label,"face_count": face_count},
            "confidence": confidence}
   