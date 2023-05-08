import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from googleapiclient.http import MediaIoBaseDownload
from keras.models import load_model

import pickle
from google.oauth2 import service_account
import io
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import urllib.request

url = 'https://github.com/pymedphys/data/releases/download/model01/marble_surface_model_final.h5'
filename = url.split('/')[-1]

urllib.request.urlretrieve(url, filename)
    
    
    
def predict(image):
    img = keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    score = tf.nn.sigmoid(prediction).numpy()[0][0]
    return score

st.title("Marble Surface Anomaly Detection")
# Create a file uploader widget in Streamlit
st.write("Upload an image of a marble surface")
image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if image is not None:
   score = predict(image)
   if score > 0.5:
      st.write("The marble surface is cracked with a probability of {}".format(score))
   else:
      st.write("The marble surface is not cracked with a probability of {}".format(1 - score))
