import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image


@st.cache(allow_output_mutation=True)
def load_model_from_drive():
    # Authenticate with Google Drive
    creds = Credentials.from_authorized_user_info(info={
        'client_id': '1049584246300-rrf03rf4hc4143o6iml77kbg1q37si8a.apps.googleusercontent.com',
        'client_secret': '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',
        
    })
    service = build('drive', 'v3', credentials=creds)

    # Download the model file from Google Drive
    file_id = 'marble_surface_model_final_1.h5'
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f'Download {int(status.progress() * 100)}.')
    fh.seek(0)

    # Load the model from the downloaded file
    model = load_model(fh)
    return model

# Load the model from Google Drive and cache it using st.cache
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
