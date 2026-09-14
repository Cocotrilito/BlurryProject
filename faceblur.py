import cv2
import numpy as np
import sys

image = cv2.imread(sys.argv[1])

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(faces)

for (x, y, w, h) in faces:
    region = image[y:y+h, x:x+w]
    region_blurred = cv2.GaussianBlur(region, (99,99), 30)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
    mask_3d = cv2.merge([mask, mask, mask])
    image[y:y+h, x:x+w] = np.where(mask_3d == 255, region_blurred, region)

cv2.imwrite("result.jpg", image)