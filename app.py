import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ===============================
# Load trained model
# ===============================
model = load_model("model.h5")

# ===============================
# Preprocessing Function
# ===============================
def preprocess_image(image):
    """
    image: numpy array from gradio (RGB)
    returns: reshaped standardized image (1,28,28,1)
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))

    # Standardize (IMPORTANT: same as training)
    resized = resized.astype("float32")
    resized = (resized - np.mean(resized)) / (np.std(resized) + 1e-7)

    # Reshape to CNN input
    reshaped = resized.reshape(1, 28, 28, 1)

    return reshaped


# ===============================
# Prediction Function
# ===============================
def predict_digit(image):
    processed = preprocess_image(image)

    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return f"Predicted Digit: {predicted_class} (Confidence: {confidence:.4f})"


# ===============================
# Gradio Interface
# ===============================
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Handwritten Digit Classifier",
    description="Draw or upload a digit (0-9). Model predicts using CNN."
)

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    interface.launch()