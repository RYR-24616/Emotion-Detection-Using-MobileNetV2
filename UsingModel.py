import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Set the paths
model_path = r"emotion_mobilenet_model.keras"
img_path = r"image.png"

# Load the model (no need for custom_objects if using .keras format)
model = load_model(model_path)

# Define class names (make sure same order as training)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load and preprocess image (RGB for MobileNet)
img = image.load_img(img_path, target_size=(48, 48))  # MobileNet expects RGB
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]

# Output
print("Predicted emotion:", predicted_class)
