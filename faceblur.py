import cv2

imagen = cv2.imread("foto.jpeg")

clasifier = cv2.CascadeClassfier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print(imagen.shape)