import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def predict_uploaded_image(model, img_path):
    img = image.load_img(img_path, target_size=(300, 300), color_mode="grayscale")
    
    # Display the image using Streamlit
    st.image(img, caption='Uploaded Image', width=300)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.

    predictions = model.predict(img_array)
    classes = ['alif', 'azar', 'diksa']
    predicted_class = classes[np.argmax(predictions)]

    return predicted_class

# Load your pre-trained model
model = load_model('./resources/your_model.h5')

st.title('Custom Image Classification with Streamlit')
st.write('Upload an image to classify')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = f"temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    predicted_class = predict_uploaded_image(model, img_path)
    st.write(f'The image is predicted as: {predicted_class}')
    
    # Remove the temporary file after prediction
    os.remove(img_path)
