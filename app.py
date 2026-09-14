import streamlit as st
import numpy as np
import cv2


st.title("Face Blur Tool")
st.write("Upload a photo and I'll blur the faces automatically")
file = st.file_uploader("Choose a photo")

from PIL import Image

if file is not None:
    image_pil = Image.open(file)
    image_array = np.array(image_pil)
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    

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
    st.image(image, channels="BGR")
    result = cv2.imencode(".png", image)[1].tobytes()
    st.download_button("Download blurred photo", result, "result.png", "image/png")
