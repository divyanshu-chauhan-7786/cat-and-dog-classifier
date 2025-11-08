import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the saved model
@st.cache_resource
def load_trained_model():
    model = load_model("cat_dog_classifier.h5")
    return model

model = load_trained_model()

# Streamlit App Title
st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image and click **Predict** to check whether it's a cat or a dog!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define prediction function
def predict_image(model, img):
    img = img.resize((128, 128)).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image(model, img)

        # Display results
        st.success(f"### Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}")
        st.markdown("---")
