import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmark.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=5)
detector = vision.FaceLandmarkerOptions.create_from_options(options)


mp_image = mp.Image.create_from_file("foto.jpeg")
result = detector.detect(mp_image)
print(result.face_landmarks)

