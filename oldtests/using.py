# predict_digit.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # convert image to 8-bit grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)  # resize the image using LANCZOS resampling
    img_array = np.array(img)
    img_array = 1 - img_array / 255.0  # normalize and invert if needed
    img_array = img_array.reshape((1, 28, 28, 1))  # reshape to match model input
    return img_array

# Function to predict the digit
def predict_digit(image_path):
    model = tf.keras.models.load_model('mnist_model.h5')
    print("Model loaded successfully")
    processed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    plt.imshow(processed_image.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.show()

# Example usage
predict_digit('num5.jpg')  # replace 'path_to_your_image.jpg' with your image path
