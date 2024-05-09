import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('letter_classification_model.h5')

# Function to preprocess images
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to match the input size of the model
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Path to new image
image_path = 'path_to_new_image.jpg'

# Preprocess the image
new_image = preprocess_image(image_path)

# Predict the image
predictions = model.predict(new_image)
predicted_class = np.argmax(predictions)

print("Predicted class:", predicted_class)
