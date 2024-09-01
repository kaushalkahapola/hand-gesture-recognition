from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the model
model = tf.keras.models.load_model("saved_model/full-set-new.keras")

# Define the image size
IMG_SIZE = 224

# Function to preprocess the image
def preprocess_image(image: Image.Image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize the image
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the file and preprocess the image
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content))
    processed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)

    # Define a mapping of class indices to gesture names
    gesture_names = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']
    predicted_gesture = gesture_names[predicted_class[0]]

    return JSONResponse(content={"predicted_gesture": predicted_gesture})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
