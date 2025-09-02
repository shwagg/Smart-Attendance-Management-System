import cv2
from facerecognition import SimpleFacerec

# encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


#pre-trained model daw na ginagamit for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        print("Capturing video failed")
        break



    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts the frame to grayscale

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    #bounding box
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    cv2.imshow("SAM: SMART ATTENDANCE MANAGEMENT SYSTEM", frame)

    #para magstop yung program
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == 27: #ASCII siya for ESC key
        break

cap.release()
cv2.destroyAllWindows()