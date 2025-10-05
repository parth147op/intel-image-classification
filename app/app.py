import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# ===============================
# Load Trained Model
# ===============================
MODEL_PATH = "app/intel_cnn_best.h5"   # adjust if needed
model = load_model(MODEL_PATH)

# Define class labels (same order as training)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Intel Image Classification", page_icon="🌍", layout="centered")

st.title("🌍 Intel Image Classification")
st.markdown("Upload a natural scene image and the model will classify it into one of six categories.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Read file → OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess for model
    img_resized = cv2.resize(img_rgb, (128,128))
    x = np.expand_dims(img_resized/255.0, axis=0)

    # Predict
    probs = model.predict(x, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx]

    # Display results
    st.image(img_rgb, caption=f"Uploaded Image", use_column_width=True)
    st.subheader(f"🔮 Prediction: **{pred_class}** ({confidence:.2%})")

    # Show probability distribution
    st.bar_chart({class_names[i]: float(probs[i]) for i in range(len(class_names))})
