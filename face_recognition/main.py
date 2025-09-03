import cv2
from facerecognition import SimpleFacerec

# Load Faces
sfr = SimpleFacerec()
sfr.load_encoding_images("dataset/")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Capturing video failed")
        break

    # Detect faces and get names
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Draw results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Smart Attendance Management System", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
