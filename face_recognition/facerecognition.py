import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25   # resize for faster recognition

    def load_encoding_images(self, images_path):
        """
        Load encoding images from the given folder.
        Each image file name should be the person's name.
        """
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print(f"{len(images_path)} encoding images found.")

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get filename as name
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            name = filename

            # Encode only first face found in the image
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(name)
            print(f"Encoding complete for {name}")

    def detect_known_faces(self, frame):
        """
        Detect known faces in a frame
        Returns: face_locations (with coordinates) and face_names
        """
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale back up face locations since the frame we detected in was scaled
        face_locations = [(int(top / self.frame_resizing), int(right / self.frame_resizing),
                           int(bottom / self.frame_resizing), int(left / self.frame_resizing)) 
                          for (top, right, bottom, left) in face_locations]

        return face_locations, face_names
