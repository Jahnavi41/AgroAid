import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load disease info JSON
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# Model paths
MODEL_PATHS = {
    "CNN (Default)": "trained_model_cnn.keras",
    "EfficientNetB3 (Transfer Learning)": "trained_model_EfficientNetB3.h5",
    "VGG16 (Transfer Learning)": "training_model_VGG16.keras",
    "ResNet50 (Transfer Learning)": "trained_model_resnet99.pth"
}

# Input sizes per model
MODEL_INPUT_SIZES = {
    "CNN (Default)": (128, 128),
    "EfficientNetB3 (Transfer Learning)": (224, 224),
    "VGG16 (Transfer Learning)": (224, 224),
    "ResNet50 (Transfer Learning)": (224, 224)
}

# Grad-CAM layer names
GRAD_CAM_LAYERS = {
    "CNN (Default)": "conv2d",
    "EfficientNetB3 (Transfer Learning)": "top_conv",
    "VGG16 (Transfer Learning)": "block5_conv3",
    "ResNet50 (Transfer Learning)": "conv5_block3_out"
}

# Class labels
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load model based on file extension
@st.cache_resource
def load_model(path):
    ext = os.path.splitext(path)[1]
    if ext in [".keras", ".h5"]:
        return tf.keras.models.load_model(path)
    elif ext == ".pth":
        return None  # Placeholder for unsupported PyTorch model
    else:
        raise ValueError("Unsupported model file type")

# Predict image class
def model_prediction(test_image, model, target_size):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions), input_arr, image

# Grad-CAM generation
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Display heatmap on image
def display_gradcam(img_array, heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + np.array(image)
    st.image(superimposed_img.astype("uint8"), caption="Grad-CAM Explanation", use_column_width=True)

# Sidebar
st.sidebar.title("🌱 AgroAid Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Evaluation"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home.jpeg", use_container_width=True)
    st.markdown("""
    Welcome to AgroAid! Upload a plant leaf image, and our AI will diagnose the disease and suggest remedies.
    Navigate to **Disease Recognition** to try it out!
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### Dataset Source
    [Kaggle - New Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

    #### Purpose
    This tool helps farmers and researchers identify crop diseases early and take action.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("📸 Upload Plant Image")
    model_choice = st.selectbox("Choose Model", list(MODEL_PATHS.keys()))
    test_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            model_path = MODEL_PATHS[model_choice]
            model = load_model(model_path)

            if model is None:
                st.error("⚠️ PyTorch (.pth) model loading not supported in this app.")
            else:
                target_size = MODEL_INPUT_SIZES[model_choice]
                result_index, confidence, input_arr, pil_img = model_prediction(test_image, model, target_size)
                predicted_class = class_name[result_index]

                st.success(f"🌿 Predicted: `{predicted_class}`")
                st.info(f"🧠 Confidence: `{confidence * 100:.2f}%`")

                if predicted_class in disease_info:
                    info = disease_info[predicted_class]
                    st.subheader("📋 Disease Info")
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Treatment:** {info['treatment']}")
                    st.write(f"**Prevention:** {info['prevention']}")

                st.subheader("🧠 Grad-CAM Explanation")
                try:
                    last_conv_layer = GRAD_CAM_LAYERS.get(model_choice, "conv2d")
                    heatmap = make_gradcam_heatmap(input_arr, model, last_conv_layer)
                    display_gradcam(input_arr, heatmap, pil_img)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate Grad-CAM. Error: {e}")

# Evaluation Page
elif app_mode == "Evaluation":
    st.header("📊 Model Evaluation")
    st.markdown("Upload a batch of images to see how well the model performs.")
    uploaded_files = st.file_uploader("Upload Multiple Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    true_labels = []
    pred_labels = []

    if uploaded_files and st.button("Evaluate"):
        model = load_model(MODEL_PATHS["CNN (Default)"])
        if model is None:
            st.error("CNN model could not be loaded.")
        else:
            target_size = MODEL_INPUT_SIZES["CNN (Default)"]
            for f in uploaded_files:
                label = f.name.split("_")[0]
                result_index, _, _, _ = model_prediction(f, model, target_size)
                true_labels.append(label)
                pred_labels.append(class_name[result_index].split("___")[0])

            st.text("Classification Report:")
            st.text(classification_report(true_labels, pred_labels))

            cm = confusion_matrix(true_labels, pred_labels, labels=list(set(true_labels)))
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(set(true_labels)), yticklabels=list(set(true_labels)))
            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(fig)

st.markdown("---")
st.caption("Built with ❤️ by AgroAid Team")
