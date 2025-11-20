# butterfly_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# ---- Page Config ----
st.set_page_config(
    page_title="Butterfly Classifier 🦋",
    page_icon="🦋",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <h1 style='text-align: center; color: #FF69B4;'>Butterfly Image Classifier 🦋</h1>
    <p style='text-align: center; color: #8B008B;'>Predict the species of a butterfly using CNN or ResNet models</p>
""", unsafe_allow_html=True)

# ---- Load Models ----
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

cnn_model = load_model("butterfly_model_deeper_100epochs.keras")
resnet_model = load_model("pretrained_model.keras")

# ---- Load class mappings ----
with open("class_indices_cnn.pkl", "rb") as f:
    cnn_class_indices = pickle.load(f)
cnn_class_names = {v: k for k, v in cnn_class_indices.items()}

with open("class_indices_resnet.pkl", "rb") as f:
    resnet_class_indices = pickle.load(f)
resnet_class_names = {v: k for k, v in resnet_class_indices.items()}

# ---- Sidebar: Model Selection ----
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", ["CNN", "ResNet"])
st.sidebar.markdown("Choose the model to use for prediction.")

# ---- Upload Image ----
st.subheader("Upload a Butterfly Image 🦋")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    img_size = 224
    img_array = np.array(image.resize((img_size, img_size))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ---- Make Prediction ----
    if model_choice == "CNN":
        preds = cnn_model.predict(img_array)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_label = cnn_class_names[pred_idx]
    else:
        preds = resnet_model.predict(img_array)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_label = resnet_class_names[pred_idx]

    # ---- Display Prediction ----
    st.success(f"Predicted Species: **{pred_label}**")

    # ---- Optional: Show prediction probabilities ----
    if st.checkbox("Show Prediction Probabilities"):
        import pandas as pd
        prob_df = pd.DataFrame(preds[0], index=(cnn_class_names if model_choice=="CNN" else resnet_class_names).values(), columns=["Probability"])
        st.dataframe(prob_df.sort_values("Probability", ascending=False).head(10))

st.markdown("""
    <hr style='border:2px solid #FF69B4;'>
    <p style='text-align: center; color: #8B008B;'>Made with ❤️ for Butterfly Lovers</p>
""", unsafe_allow_html=True)
