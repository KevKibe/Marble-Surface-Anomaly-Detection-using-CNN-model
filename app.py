import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load the model
def load_model():
    model=('model_surface_model_2.json')
    return model

#Caching the model for faster loading
@st.cache

def predict(image):
    img = keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    score = tf.nn.sigmoid(prediction).numpy()[0][0]
    return score

st.title("Marble Surface Anomaly Detection")
# Create a file uploader widget in Streamlit
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
st.title("Marble Surface Classification")
st.write("Upload an image of a marble surface")
image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if image is not None:
   score = predict(image)
   if score > 0.5:
      st.write("The marble surface is cracked with a probability of {}".format(score))
   else:
      st.write("The marble surface is not cracked with a probability of {}".format(1 - score))
