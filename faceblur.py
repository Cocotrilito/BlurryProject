import cv2

image = cv2.imread("foto.jpeg")

clasifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = clasifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(faces)

for (x, y, w, h) in faces:
    region = image[y:y+h, x:x+w]
    region_blurred = cv2.GaussianBlur(region, (99,99), 30)
    image[y:y+h, x:x+w] = region_blurred

cv2.imwrite("result.jpg", image)