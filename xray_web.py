import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np


# Load the model without compiling (fixes loss function issues with .h5)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/nndl.keras", compile=False)
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.title("🩺 Pneumonia Identification System")
st.markdown("Upload a chest X-ray image to predict if pneumonia is present.")

file = st.file_uploader("Upload a chest scan", type=["jpg", "jpeg", "png"])

# Prediction function
def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, method=Image.Resampling.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# Define class names based on your model output
class_names = ['Normal', 'Pneumonia']

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Predicting...'):
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])

        st.subheader("Prediction Result:")
        st.write(f"**Prediction:** {class_names[np.argmax(score)]}")
