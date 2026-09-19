import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import io




if "blur_faces" not in st.session_state:
    st.session_state.blur_faces = {}




st.title("Face Blur Tool")
st.write("Upload a photo and I'll blur the faces automatically")
file = st.file_uploader("Choose a photo")


FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                      397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                      172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


if file is not None:
    image_pil = Image.open(file)
    image_array = np.array(image_pil)
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=5)
    detector = vision.FaceLandmarker.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    result = detector.detect(mp_image)

    if len(result.face_landmarks) == 0:
        st.warning("Ups.. No faces detected in this photo!")
    else:
        with st.spinner("Processing hold on..."):

            height, width, _ = image.shape
            mask = np.zeros((height, width), dtype=np.uint8)

            image_display = image.copy()

            

            face_boxes = []
            for i, face_landmarks in enumerate(result.face_landmarks):
                xs = [int(p.x * width) for p in face_landmarks]
                ys = [int(p.y * height) for p in face_landmarks]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                face_boxes.append((x_min, y_min, x_max, y_max))

            if "manual_boxes" not in st.session_state:
                st.session_state.manual_boxes = []
            if "first_corner" not in st.session_state:
                st.session_state.first_corner = None

            image_display = image.copy()
            for i, (x_min, y_min, x_max, y_max) in enumerate(face_boxes):
                color = (0, 255, 0) if st.session_state.blur_faces.get(i, True) else (0, 0, 255)
                cv2.rectangle(image_display, (x_min, y_min), (x_max, y_max), color, 3)

            mode = st.radio("Click mode:", ["Toggle Face blur", "Draw manual rectangle"])

            click = streamlit_image_coordinates(
                cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB),
                key="unified_selector",
                width=600
            )
           

            if click is not None:
                click_key = click["unix_time"]
                if st.session_state.get("last_click") != click_key:
                    st.session_state.last_click = click_key
                    scale = width /600
                    point = (int(click["x"] * scale), int(click["y"] * scale))

                    print("Debug - point:", point, "mode:", mode, "first_corner:", st.session_state.first_corner)


                    if mode == "Toggle Face blur":
                        click_x, click_y = point
                        for i, (x_min, y_min, x_max, y_max) in enumerate(face_boxes):
                            
                            if x_min <= click_x <= x_max and y_min <= click_y <= y_max:
                                current = st.session_state.blur_faces.get(i, True)
                                st.session_state.blur_faces[i] = not current
                        st.rerun()
                    else:
                        if st.session_state.first_corner is None:
                            st.session_state.first_corner = point
                        else:
                            x1, y1 = st.session_state.first_corner
                            x2, y2 = point
                            new_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                            st.session_state.manual_boxes.append(new_rect)
                            st.session_state.first_corner = None
                        st.rerun()




            for i, face_landmarks in enumerate(result.face_landmarks):
                if st.session_state.blur_faces.get(i, True):
                    contour_points = []
                    for index in FACE_OVAL_INDICES:
                        point = face_landmarks[index]
                        x = int(point.x * width)
                        y = int(point.y * height)
                        contour_points.append([x, y])
                    puntos_array = np.array(contour_points, dtype=np.int32)
                    cv2.fillPoly(mask, [puntos_array], 255)


        


        blur_style = st.selectbox("Censorship Style", ["Blur", "Pixelate", "Black Bar"])
        blur_intensity = st.slider("Blur Intensity", min_value=15, max_value=151, value=99, step=2)




        if blur_style == "Blur":
            image_censored = cv2.GaussianBlur(image, (blur_intensity, blur_intensity), 30)
        elif blur_style == "Pixelate":
            pixel_size = max(2, blur_intensity // 10)
            small = cv2.resize(image, (width // pixel_size, height // pixel_size))
            image_censored = cv2. resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        elif blur_style == "Black Bar":
            opacity = blur_intensity / 151
            black= np.zeros_like(image)
            image_censored = cv2.addWeighted(image, 1 - opacity, black, opacity, 0)


        mask_3d = cv2.merge([mask, mask, mask])
        for (x_min, y_min, x_max, y_max) in st.session_state.manual_boxes:
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
        result = np.where(mask_3d == 255, image_censored, image)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original")
            st.image(image, channels="BGR")
        with col2:
            st.write("Blurred")
            st.image(result, channels="BGR")
        resultBytes = cv2.imencode(".png", result)[1].tobytes()
        st.download_button("Download blurred photo", resultBytes, "result.png", "image/png")
