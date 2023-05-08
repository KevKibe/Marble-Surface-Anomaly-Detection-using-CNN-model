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

#@st.cache_resource
def load_model_from_drive():
    # Replace these variables with your own values
    SERVICE_ACCOUNT_FILE = 'peerless-dahlia-385616-ba2dd85ba63a.json'
    FILE_ID = 'marble_surface_model_final_1.h5'

    # Authenticate with the service account
    credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly'])

    # Build the Drive API client
    service = build('drive', 'v3', credentials=credentials)
    try:
        # Retrieve the file from Google Drive
        request = service.files().get_media(fileId=FILE_ID)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f'Download {int(status.progress() * 100)}.')

        # Load the model from the downloaded file
        file.seek(0)
        model = load_model(file)

        return model
    except HttpError as error:
        print(f'An error occurred: {error}')

# Load the model using the cached function
model = load_model_from_drive()
    
    
    
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
