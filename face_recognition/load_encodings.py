import pickle

def load_face_encodings(file_path="encodings.dat"):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]
    except FileNotFoundError:
        return [], []