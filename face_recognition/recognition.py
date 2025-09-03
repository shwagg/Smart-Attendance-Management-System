import face_recognition
import pickle
import os

known_face_encodings = []
known_face_names = []
dataset_path = 'dataset'

for name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, name)
    if os.path.isdir(person_dir):
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

# Save the encodings to a file
with open("encodings.dat", "wb") as f:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)