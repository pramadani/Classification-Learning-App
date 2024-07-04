import streamlit as st
import cv2
import numpy as np
from dashboard import loaded_model
from tensorflow.keras.preprocessing import image

# Load your Keras model for face recognition
# loaded_model = load_model('./resources/model.h5')

# Load pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of class names (adjust according to your model)
classes = ['alif', 'azar', 'diksa']

st.title('Face Recognition from Custom Image')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_real = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # IMREAD_COLOR is equivalent to 1
    image_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)
    image_real = cv2.resize(image_real, (300, 300))

    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize the image for better processing
    gray = cv2.resize(image, (800, 600))  # Adjust size as needed

    # Convert to grayscale
    # gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        st.write("No faces detected. Using the whole image for prediction.")
        face_gray_resized = cv2.resize(gray, (300, 300))  # Resize grayscale image
        img_array = np.expand_dims(face_gray_resized, axis=0) / 255.

        # Perform prediction with your loaded model
        predictions = loaded_model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Display the whole image and predicted name
        st.image(image_real, channels="RGB", caption=f"Real Image")
        st.write("Preprocessing Image")
        st.image(face_gray_resized, channels="G", caption=f"Predicted: {predicted_class} ({confidence:.2f})")

        # Optionally, you can display more information or save the results
        # st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

    else:
        # Process each detected face
        for (x, y, w, h) in faces:
            # Crop the face
            face = gray[y:y+h, x:x+w]

            # Convert face to grayscale
            # face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Preprocess the face image for your model
            face_gray_resized = cv2.resize(face, (300, 300))
            img_array = np.expand_dims(face_gray_resized, axis=0) / 255.

            # Perform prediction with your loaded model
            predictions = loaded_model.predict(img_array)
            predicted_class = classes[np.argmax(predictions)]
            confidence = np.max(predictions)

            # Display the cropped face and predicted name
            st.image(image_real, channels="RGB", caption=f"Real Image")
            st.write("Preprocessing Image")
            st.image(face_gray_resized, channels="G", caption=f"Predicted: {predicted_class} ({confidence:.2f})")

            # Optionally, you can display more information or save the results
            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}")

else:
    st.write("Upload an image to get started.")
