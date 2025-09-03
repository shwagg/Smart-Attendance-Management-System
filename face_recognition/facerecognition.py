import face_recognition
import cv2
import os
import glob
import numpy as np
import pickle

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25   # resize for faster recognition

    def load_encoding_images(self, images_path="dataset/"):
        """
        Load encoding images from the dataset folder.
        Each subfolder name is the person's name,
        and multiple images per person are supported.
        """
        subfolders = [os.path.join(images_path, f) for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]

        for folder in subfolders:
            name = os.path.basename(folder)
            images = glob.glob(os.path.join(folder, "*.*"))

            print(f"{len(images)} images found for {name}.")

            for img_path in images:
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(rgb_img)
                if len(encodings) > 0:
                    img_encoding = encodings[0]
                    self.known_face_encodings.append(img_encoding)
                    self.known_face_names.append(name)
                    print(f"Encoding complete for {name} - {os.path.basename(img_path)}")
                else:
                    print(f"No face found in {img_path}, skipping.")
            if name not in self.known_face_names:
                print(f"No valid encodings found for {name}, check dataset images.")

        with open("encodings.dat", "wb") as f:
            pickle.dump({
                "encodings": self.known_face_encodings,
                "names": self.known_face_names
            }, f)
        print("Encodings saved to encodings.dat")
        


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

        face_locations = [(int(top / self.frame_resizing), int(right / self.frame_resizing),
                           int(bottom / self.frame_resizing), int(left / self.frame_resizing)) 
                          for (top, right, bottom, left) in face_locations]

        return face_locations, face_names
