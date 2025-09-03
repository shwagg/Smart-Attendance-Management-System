import cv2
import face_recognition
from load_encodings import load_face_encodings

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier('face_recognition\\haarcascades\\haarcascade_frontalface_default.xml')# Load the face encodings and names from your saved file
known_face_encodings, known_face_names = load_face_encodings()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Capturing video failed")
        break

    # Convert the frame to grayscale for Haar Cascade (Face Detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Convert the frame to RGB for face_recognition library (Face Recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face locations and encodings for the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a bounding box and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Smart Attendance Management System", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == 27:  # 27 is the ASCII for the ESC key
        break

cap.release()
cv2.destroyAllWindows()