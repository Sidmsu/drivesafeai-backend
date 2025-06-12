import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def preprocess_image(image_path, target_size=(640, 480)):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1000:
        image = cv2.resize(image, target_size)
    return image

def compute_ear(image_path):
    image = preprocess_image(image_path)
    height, width = image.shape[:2]

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]

        left_eye = np.array([[int(face_landmarks.landmark[i].x * width),
                              int(face_landmarks.landmark[i].y * height)] for i in LEFT_EYE_IDX])
        right_eye = np.array([[int(face_landmarks.landmark[i].x * width),
                               int(face_landmarks.landmark[i].y * height)] for i in RIGHT_EYE_IDX])

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = round((left_ear + right_ear) / 2.0, 2)

        return round(1.0 - avg_ear, 2), round(avg_ear, 2)
