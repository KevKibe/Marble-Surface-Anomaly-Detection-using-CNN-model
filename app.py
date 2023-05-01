import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('model_surface_model_2.json')
@st.cache
# Define the image size and target labels
img_width, img_height = 224, 224
labels = {0: 'Defective', 1: 'Good'}

st.title("Marble Surface Anomaly Detection")
# Create a file uploader widget in Streamlit
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    img = image.load_img(uploaded_file, target_size=(img_width, img_height))
    # Preprocess the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    # Make a prediction
    prediction = model.predict(img)
    # Get the predicted label
    predicted_label = labels[np.argmax(prediction)]
    # Rewrite the output as 'Defective' for class 0 and 'Good' for class 1
    if predicted_label == '0':
        predicted_label = 'Defective'
    elif predicted_label == '1':
        predicted_label = 'Good'
    # Display the predicted label in Streamlit
    st.write("The image is classified as:", predicted_label)
