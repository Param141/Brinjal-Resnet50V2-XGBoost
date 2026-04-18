import streamlit as st
import tensorflow as tf
import xgboost as xgb
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# =========================================================
# App Configuration
# =========================================================
st.set_page_config(
    page_title="Hybrid Brinjal Disease Detection",
    page_icon="🍆",
    layout="wide"
)

st.title("🍆 Hybrid AI Brinjal Disease Detection (ResNet + XGBoost)")
st.markdown("Upload a Brinjal leaf image. This app uses a CNN to extract features and an XGBoost algorithm to classify the disease.")

# =========================================================
# Load BOTH Trained Artifacts
# =========================================================
EXTRACTOR_PATH = "resnet50v2_extractor_3class.h5"
XGB_PATH = "xgb_classifier_3class.json"
IMG_SIZE = (224, 224)

CLASS_NAMES = ['Healthy Leaves', 'Little Leaf', 'Phomopsis Blight']

@st.cache_resource
def load_hybrid_models():
    # 1. Load Keras Feature Extractor
    extractor = tf.keras.models.load_model(EXTRACTOR_PATH)
    
    # 2. Load XGBoost Classifier
    classifier = xgb.XGBClassifier()
    classifier.load_model(XGB_PATH)
    
    return extractor, classifier

# Load models into memory
try:
    feature_extractor, xgb_classifier = load_hybrid_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False

# =========================================================
# Image Upload & Prediction UI
# =========================================================
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("#### Diagnosis Results")
        if models_loaded:
            with st.spinner("Running Hybrid Pipeline (CNN -> XGBoost)..."):
                
                # --- PREPROCESSING ---
                image_rgb = image.convert('RGB')
                img_resized = image_rgb.resize(IMG_SIZE)
                
                # CRITICAL: Convert to float32 for ResNet50V2
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)
                img_processed = preprocess_input(img_array)
                
                # --- PHASE 1: CNN FEATURE EXTRACTION ---
                # Extracts 2048 mathematical features from the image
                features = feature_extractor.predict(img_processed)
                
                # --- PHASE 2: XGBOOST CLASSIFICATION ---
                # Predict probabilities based on the 2048 features
                xgb_probabilities = xgb_classifier.predict_proba(features)[0]
                
                # Find the highest probability
                predicted_class_index = np.argmax(xgb_probabilities)
                max_confidence = float(xgb_probabilities[predicted_class_index] * 100)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                
                # --- UI LOGIC (Only show if > 50% confident) ---
                if max_confidence > 50.0:
                    st.success(f"**Diagnosis:** {predicted_class_name}")
                    st.caption(f"The hybrid model is highly confident in this result.")
                else:
                    st.warning("The model is unsure about this image. Please upload a clearer, well-lit photo of the leaf.")
                    
                # Optional: Expandable section for debugging/nerds
                with st.expander("View Raw Model Probabilities"):
                    for idx, class_name in enumerate(CLASS_NAMES):
                        st.write(f"- {class_name}: {xgb_probabilities[idx]*100:.2f}%")