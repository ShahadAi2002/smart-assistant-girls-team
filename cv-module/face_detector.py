import cv2
def detect_faces(image_path):
    # read the image
    img = cv2.imread(image_path)
    #convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Loading Haar Cascade Classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=7, minSize=(50, 50))

    #drawing bounding boxes
    for (x, y, w, h) in face:
       cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2) 

    return len(face), face   


