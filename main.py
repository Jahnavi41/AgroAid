import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import pandas as pd
from io import BytesIO
from PIL import Image

# Load disease info JSON
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# Model paths
MODEL_PATHS = {
    "CNN (Default)": "trained_model_cnn.keras",
    "EfficientNetB3 (Transfer Learning)": "trained_model_EfficientNetB3.h5",
    "VGG16 (Transfer Learning)": "training_model_VGG16.keras",
    "ResNet50 (Transfer Learning)": "trained_model_resnet99.pth"  # Skipped in evaluation
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
class_name = [  # trimmed for brevity if needed
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

# Load model
@st.cache_resource
def load_model(path):
    ext = os.path.splitext(path)[1]
    if ext in [".keras", ".h5"]:
        return tf.keras.models.load_model(path)
    elif ext == ".pth":
        return None
    else:
        raise ValueError("Unsupported model file type")

# Predict
def model_prediction(test_image, model, target_size):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions), input_arr, image

# Grad-CAM
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

# Grad-CAM overlay
def display_gradcam(img_array, heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + np.array(image)
    st.image(superimposed_img.astype("uint8"), caption="Grad-CAM Explanation", use_column_width=True)

# Sidebar
st.sidebar.title("🌱 AgroAid Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Evaluation", "Model Comparison"])

# Home
if app_mode == "Home":
    st.header("🌾 Plant Disease Recognition System")
    st.image("home.jpeg", use_container_width=True)
    st.markdown("Upload a plant leaf image, and our AI will detect disease and suggest treatment.")

# About
elif app_mode == "About":
    st.header("📖 About AgroAid")
    st.markdown("""
    **Dataset:** [Kaggle - New Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  
    **Purpose:** Assist farmers with early disease detection and intervention using AI.  
    """)

# Recognition
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
                st.error("⚠️ PyTorch (.pth) model not supported yet.")
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
                    st.warning(f"⚠️ Grad-CAM failed: {e}")

# Evaluation
elif app_mode == "Evaluation":
    st.header("📊 Model Evaluation")
    st.markdown("Upload test images (filenames must include true label)")

    uploaded_files = st.file_uploader("Upload Multiple Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if uploaded_files and st.button("Evaluate All Models"):
        valid_models = [m for m in MODEL_PATHS if not MODEL_PATHS[m].endswith(".pth")]
        tabs = st.tabs(valid_models)

        for i, model_name in enumerate(valid_models):
            with tabs[i]:
                st.subheader(f"🔍 Evaluating {model_name}")
                model = load_model(MODEL_PATHS[model_name])
                target_size = MODEL_INPUT_SIZES[model_name]

                true_labels = []
                pred_labels = []

                for f in uploaded_files:
                    true_label = f.name.split("_")[0]
                    pred_idx, _, _, _ = model_prediction(f, model, target_size)
                    predicted_class = class_name[pred_idx].split("___")[0]
                    true_labels.append(true_label)
                    pred_labels.append(predicted_class)

                # Report
                st.text("Classification Report:")
                st.code(classification_report(true_labels, pred_labels), language='text')

                # Confusion matrix
                st.markdown("### Confusion Matrix")
                labels = sorted(list(set(true_labels)))
                cm = confusion_matrix(true_labels, pred_labels, labels=labels)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"Confusion Matrix for {model_name}")
                st.pyplot(fig)
# Add this inside your Streamlit app under a new page: "Model Comparison"

elif app_mode == "Model Comparison":
    st.header("🤖 Model Comparison")
    selected_models = st.multiselect("Select Models to Compare", list(MODEL_PATHS.keys()), default=list(MODEL_PATHS.keys())[:2])
    uploaded_files = st.file_uploader("Upload Images for Comparison", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files and selected_models:
        results = []

        st.subheader("📋 Prediction Results")
        progress = st.progress(0)
        total_ops = len(uploaded_files) * len(selected_models)
        op_count = 0

        for img_file in uploaded_files:
            img_name = img_file.name
            st.markdown(f"**Image:** `{img_name}`")
            img_display = st.image(img_file, width=200)

            row_data = {"Image": img_name}

            for model_name in selected_models:
                model_path = MODEL_PATHS[model_name]
                model = load_model(model_path)
                if model is None:
                    row_data[model_name] = "(PyTorch model not supported)"
                    continue

                target_size = MODEL_INPUT_SIZES[model_name]
                result_index, confidence, input_arr, pil_img = model_prediction(img_file, model, target_size)
                pred_label = class_name[result_index]
                row_data[model_name] = f"{pred_label} ({confidence * 100:.2f}%)"
                op_count += 1
                progress.progress(op_count / total_ops)

            results.append(row_data)

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        csv = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", data=csv, file_name="model_comparison_results.csv", mime="text/csv")

        # Bar Chart for visual analysis
        st.subheader("📊 Prediction Confidence Comparison")
        for row in results:
            img_name = row["Image"]
            st.markdown(f"#### 📸 `{img_name}`")
            confidences = []
            labels = []
            for model_name in selected_models:
                value = row[model_name]
                if "(" in value:
                    label, conf = value.rsplit("(", 1)
                    conf = float(conf.replace("%)", ""))
                    confidences.append(conf)
                    labels.append(model_name + ": " + label.strip())
            fig, ax = plt.subplots()
            ax.barh(labels, confidences, color="skyblue")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Confidence Comparison")
            st.pyplot(fig)

        # Optional Grad-CAM visuals for one image and all models
        st.subheader("🧠 Grad-CAM Visualization (First Image)")
        first_image = uploaded_files[0]
        for model_name in selected_models:
            st.markdown(f"##### 🔍 {model_name}")
            model_path = MODEL_PATHS[model_name]
            model = load_model(model_path)
            if model is None:
                st.warning("PyTorch models not supported for Grad-CAM")
                continue
            target_size = MODEL_INPUT_SIZES[model_name]
            result_index, confidence, input_arr, pil_img = model_prediction(first_image, model, target_size)
            try:
                heatmap = make_gradcam_heatmap(input_arr, model, GRAD_CAM_LAYERS[model_name])
                display_gradcam(input_arr, heatmap, pil_img)
            except Exception as e:
                st.warning(f"Grad-CAM failed for {model_name}: {e}")


# Footer
st.markdown("---")
st.caption("🚀 Built with ❤️ by AgroAid Team")
