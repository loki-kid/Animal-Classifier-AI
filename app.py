import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

MODEL_PATH = "animal_model.h5"   
DATASET_PATH = "dataset"         
IMG_SIZE = (224, 224)


st.set_page_config(page_title="Animal Classifier", layout="centered")

st.title("Animal Classifier AI")
st.write("Upload ảnh động vật để AI dự đoán")


@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])


uploaded_file = st.file_uploader(
    "Chọn ảnh", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Ảnh đã tải lên", width=300)


    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)


    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction)


    st.subheader("Kết quả dự đoán:")
    st.write(f"**Loài:** {predicted_class}")
    st.write(f"**Độ tin cậy:** {confidence:.2f}")

    st.subheader("Xác suất từng class:")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {prediction[0][i]:.2f}")