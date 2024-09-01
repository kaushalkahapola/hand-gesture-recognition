# Hand Gesture Recognition

Hand Gesture Recognition app using FastAPI and Streamlit for real-time gesture prediction. The backend uses a TensorFlow model to predict hand gestures from uploaded or captured images.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
  
## Overview

This project aims to recognize hand gestures using images uploaded by the user or captured from the camera. The model used for prediction is a TensorFlow model trained on the 'Hand Gesture Recognition Database' from Kaggle. The application includes a FastAPI backend for handling image processing and a Streamlit frontend for user interaction.

## Features

- Upload images for gesture recognition.
- Predictions are made using a custom-trained TensorFlow model.
  
## Dataset

The dataset used for training the model is the **Hand Gesture Recognition Database** available on Kaggle. This dataset contains near-infrared images acquired by the Leap Motion sensor, consisting of 10 different gestures performed by 10 different subjects. The dataset is organized into folders for each subject and gesture.

- **Gestures**: palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down
- **Color Channel**: Grayscale (1 channel)
- **Link**: https://www.kaggle.com/datasets/gti-upm/leapgestrecog/data

## Installation

### FastAPI Backend

1. Clone the repository:
    ```bash
    git clone https://github.com/kaushalkahapola/hand-gesture-recognition.git
    cd hand-gesture-recognition/hand_gesture_api
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the pre-trained model and place it in the `saved_model/` directory:
    [Download the model](https://drive.google.com/uc?export=download&id=1-6MVxCI5GUHu05AWo_Q7sofBfx83kw33)

### Streamlit Frontend

1. Navigate to the Streamlit app directory:
    ```bash
    cd ../hand_gesture_streamlit
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### FastAPI Backend

1. Start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```

### Streamlit Frontend

1. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to access the Streamlit interface.


## Screenshots

![image](https://github.com/user-attachments/assets/401bfd7a-c386-42e8-8d0f-51575a87324a)

![image](https://github.com/user-attachments/assets/63b92bfe-9f17-4617-b4bb-767c4e0a8d01)


