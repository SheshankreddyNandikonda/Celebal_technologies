import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load Trained Model & Encoder
clf = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Load ResNet50 model
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Feature Extraction Function
def extract_features_from_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    features = resnet.predict(image_array, verbose=0)
    return features.flatten()

# Streamlit UI
st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("Image Classifier using RandomForest")
st.markdown("Upload an image, and this model will classify it using features extraction.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Extracting features and predicting..."):
        features = extract_features_from_image(image)
        prediction = clf.predict([features])
        predicted_label = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Class: **{predicted_label}**")
