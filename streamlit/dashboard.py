import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.title("Face Recognition from Webcam")

# Load your Keras model for face recognition
loaded_model = load_model('./resources/model.h5')

# Access the webcam (use 0 for default camera, 1 or higher for external cameras)
cap = cv2.VideoCapture(4)

stframe = st.empty()
stframe_faces = st.empty()

# Load pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List of class names (adjust according to your model)
classes = ['alif', 'azar', 'diksa']

# Loop to continuously get frames from the webcam
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        st.write("Failed to capture image")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the faces and put the predicted name next to the rectangle
    face_frames = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face image for your model
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray_resized = cv2.resize(face_gray, (300, 300))
        img_array = np.expand_dims(face_gray_resized, axis=0) / 255.
        
        # Perform prediction with your loaded model
        predictions = loaded_model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]
        
        # Prepare the text to display (predicted name)
        text = f"{predicted_class} ({np.max(predictions):.2f})"
        
        # Put the text next to the rectangle
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        face_gray_resized = cv2.resize(face_gray, (100, 100))
        face_frames.append((face_gray_resized, text))

    # Display the resulting frame
    stframe.image(frame, channels="BGR")

    if face_frames:
        # Create a horizontal stack of cropped faces and names
        faces_display = []
        for face, text in face_frames:
            faces_display.append(face)
        faces_display = np.hstack(faces_display)
        
        stframe_faces.image(faces_display, channels="GRAY")
    else:
        stframe_faces.image(np.zeros((100, 100)), channels="GRAY")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
