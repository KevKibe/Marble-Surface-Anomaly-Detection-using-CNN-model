import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image


model = tf.keras.models.load_model('model/marble_class_model.h5')


class_names = ['defect','good']
target_size = (48, 48)

def preprocess_image(image, target_size):
    # Resize and normalize the image
    img = image.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

def predict(image):
    # Preprocess the image
    img = preprocess_image(image, target_size)
    
    # Make prediction
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

st.title("Marble Surface Anomaly Detection")
# Create a file uploader widget in Streamlit
st.write("Upload an image of a marble surface")
image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if image is not None:
    # Read the image
    img = Image.open(image)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    prediction = predict(img)
    st.write("Prediction:", prediction)
