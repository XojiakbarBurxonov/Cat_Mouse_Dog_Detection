import streamlit as slt # type: ignore
from fastai.vision.all import * # type: ignore
from PIL import Image # type: ignore

import io

# Title
slt.title("Cat Mouse Dog Detection")

# Upload image
file = slt.file_uploader("Choose image...", type=['jpeg', 'png', 'gif', 'svg'])

# Function to convert the uploaded file to a PIL Image
def convert_uploaded_file_to_pil_image(uploaded_file):
    if uploaded_file:
        try:
            # Read the file into a byte stream and open it as an image
            image = Image.open(io.BytesIO(uploaded_file.read()))
            return image
        except Exception as e:
            slt.error(f"Error processing the image file: {e}")
    return None

# Convert the uploaded file to a PIL Image
img = convert_uploaded_file_to_pil_image(file)

if img:
    # Display the uploaded image
    slt.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Load the model
    model = load_learner('cat_mouse_dog_detection.pkl') # type: ignore
    
    # Prediction
    pred, pred_idx, probs = model.predict(img)
    
    # Display results
    slt.write(f"Prediction: {pred}")
    slt.write(f"Probability: {probs[pred_idx]:.4f}")
else:
    slt.write("Upload an image to see the prediction.")
