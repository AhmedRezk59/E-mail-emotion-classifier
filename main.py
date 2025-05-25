import streamlit as st
from PIL import Image
import joblib
import os

def classify_email(text):
    model = joblib.load("models/best_model.pkl")
    # Ensure input is in the correct format for the model
    if hasattr(model, "predict"):
        import pandas as pd
        input_df = pd.DataFrame({'text': [text]})
        prediction = model.predict(input_df)
        return prediction[0]  # Return the predicted label (e.g., "happy", "sad", etc.)
    else:
        raise ValueError("Loaded model does not have a predict method.")
   

# Image mapping (replace with your own images)
IMAGE_PATHS = {
    "happy": "images/happy_image.gif",
    "sad": "images/sad_image.gif",
    "angry": "images/angry_image.gif",
    "neutral": "images/neutral_image.gif"
}

st.title("Email Emotion Classifier")

email_text = st.text_area("Enter your email text:")

if st.button("Classify"):
    emotion = classify_email(email_text)
    st.write(f"Detected emotion: **{emotion.capitalize()}**")
    image_path = os.path.join(IMAGE_PATHS.get(emotion, IMAGE_PATHS["neutral"]))
    try:
        img = Image.open(image_path)
        st.image(img, caption=emotion.capitalize())
    except FileNotFoundError:
        st.warning(f"Image for '{emotion}' not found.")