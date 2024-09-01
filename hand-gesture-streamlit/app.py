import streamlit as st
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit.components.v1 as components

st.title("Hand Gesture Recognition")

# Define the FastAPI backend URL
backend = "http://127.0.0.1:8000/predict/"

def process(image, server_url: str):
    # Create a MultipartEncoder object
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    # Send the POST request
    response = requests.post(
        server_url,
        data=m,
        headers={"Content-Type": m.content_type},
        timeout=8000
    )
    return response

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Process the image with FastAPI
    response = process(uploaded_file, backend)

    # Display the prediction result
    if response.status_code == 200:
        result = response.json()
        st.write(f"Predicted Gesture: {result['predicted_gesture']}")
    else:
        st.write("Error in prediction")

