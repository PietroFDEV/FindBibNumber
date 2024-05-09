import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def load_and_preprocess_image(image_path, image_width=56):
    """Load an image, convert it to grayscale, resize it and normalize."""
    img = Image.open(image_path).convert('L')  # Convert image to 8-bit grayscale
    img = img.resize((image_width, 28), Image.Resampling.LANCZOS)  # Resize the image
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to 0-1
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array

def predict_digits(model_path, image_path):
    """Load the trained model and predict the digits in the image."""
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    processed_image = load_and_preprocess_image(image_path)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    prediction = model.predict(processed_image)
    predicted_number = np.argmax(prediction)

    plt.imshow(processed_image.reshape(28, 56), cmap='gray')
    plt.title(f'Predicted Number: {predicted_number}')
    plt.show()

# Example usage
predict_digits('multi_digit_model.h5', 'nums.jpg')  # Replace with your image path
