import os
import tensorflow as tf
from keras.models import load_model
import streamlit as st
import numpy as np

st.header('Flower Classification Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load your trained model
model = load_model('flower_reg_model.h5')

# Classification function
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + \
              ' with a score of ' + str(np.max(result)*100) + '%'
    return outcome

# File uploader
uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    # Ensure the upload folder exists
    upload_folder = 'upload'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, uploaded_file.name)

    # Save the uploaded file
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the image
    st.image(uploaded_file, width=200)

    # Display classification result
    st.markdown(classify_images(file_path))
