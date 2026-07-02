import cv2
def detect_faces(image_path):
    # read the image
    img = cv2.imread(image_path)
    #convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Loading face Haar Cascade Classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

    return len(face), face   

def draw_faces(image_path, output_path):
   # read the image
   img = cv2.imread(image_path)
   
   #call detect_faces
   __,face = detect_faces(image_path)

   #drawing bounding boxes
   for (x, y, w, h) in face:
       cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2) 
   
   #save img
   cv2.imwrite(output_path, img)   

def detect_faces_with_eyes(image_path):
     # read the image
    img = cv2.imread(image_path)
    #convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_count, faces = detect_faces(image_path)

    #Loading eye Haar Cascade Classifiers
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    eyes_detected = []

    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=5)
        eyes_detected.append(len(eyes))

    return eyes_detected
