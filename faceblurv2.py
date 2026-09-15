import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np




base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=5)
detector = vision.FaceLandmarker.create_from_options(options)


mp_image = mp.Image.create_from_file("foto.jpeg")
result = detector.detect(mp_image)

FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]



face_landmarks = result.face_landmarks[0]

image = cv2.imread("foto.jpeg")
height, width, _ = image.shape

contour_points = []
for index in FACE_OVAL_INDICES:
    point = face_landmarks[index]
    x = int(point.x * width)
    y = int(point.y * height)
    contour_points.append([x, y])
print(contour_points)

mask = np.zeros((height, width), dtype=np.uint8)
array_points = np.array(contour_points, dtype=np.int32)
cv2.fillPoly(mask, [array_points], 255)


image_blurred = cv2.GaussianBlur(image, (99, 99), 30)
mask_3d = cv2.merge([mask, mask, mask])
final_result = np.where(mask_3d == 255, image_blurred, image)

cv2.imwrite("result_v2.jpg", final_result)
