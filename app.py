import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Surface Crack Detection", page_icon="🔎", layout="wide")

IMAGE_SIZE = 128
MODEL_PATH = "Final_Crack_Detection_Model.keras"

st.title("🔎 Industrial Surface Crack Detection")
st.write("Upload a surface image to check whether the CNN model classifies it as **Crack Detected** or **No Crack**.")

@st.cache_resource
def load_crack_model():
    return load_model(MODEL_PATH)

with st.sidebar:
    st.header("About the Project")
    st.write("**Model:** Convolutional Neural Network (CNN)")
    st.write("**Task:** Binary image classification")
    st.write("**Input size:** 128 × 128 pixels")
    st.caption("Student project demonstration only. Do not use this prediction as the sole basis for industrial safety decisions.")

try:
    model = load_crack_model()
except Exception as error:
    st.error(f"Could not load `{MODEL_PATH}`. Put the trained model in the same folder as app.py.")
    st.code(str(error))
    st.stop()

uploaded_file = st.file_uploader("Upload a surface image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns([1, 1.2], gap="large")

        with left:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        with right:
            st.subheader("Prediction")
            resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing image..."):
                score = float(model.predict(image_array, verbose=0)[0][0])

            # Confirm this mapping against the Class Indices printed by training.
            # With folder names Crack and NoCrack, Keras usually assigns Crack=0, NoCrack=1.
            crack_score = 1.0 - score
            no_crack_score = score

            if crack_score >= no_crack_score:
                result, confidence = "Crack Detected", crack_score
                st.error(f"⚠️ {result}")
            else:
                result, confidence = "No Crack", no_crack_score
                st.success(f"✅ {result}")

            st.metric("Confidence for predicted class", f"{confidence * 100:.2f}%")
            st.progress(float(np.clip(confidence, 0.0, 1.0)))

            with st.expander("View model scores"):
                st.write(f"Crack score: {crack_score * 100:.2f}%")
                st.write(f"No Crack score: {no_crack_score * 100:.2f}%")

            st.info("Results depend on image quality and the training data. Verify findings independently.")
    except Exception as error:
        st.error("The image could not be processed.")
        st.code(str(error))
else:
    st.info("Upload an image to start the prediction.")
