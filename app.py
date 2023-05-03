import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from keras.models import load_model

import pickle
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

@st.cache(show_spinner=False)
def load_model_from_drive():
    # Authenticate and authorize the Google Drive API credentials
    creds = Credentials.from_authorized_user_file('/path/to/credentials.json', ['https://www.googleapis.com/auth/drive'])

    # Create a connection to your Google Drive account using the authenticated credentials
    service = build('drive', 'v3', credentials=creds)

    # Fetch the model file from the Google Drive
    file_id = 'your_file_id'
    request = service.files().get_media(fileId=file_id)
    file = request.execute()

    # Load the model from the fetched file using pickle
    model = pickle.loads(file)

    return model

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
