import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def load_and_prepare_image(image_path):
    # Load the image
    img = Image.open(image_path)
    img = img.convert('L')  # Convert to grayscale
    
    # Resize to 28x28 pixels
    img = img.resize((28, 28), Image.ANTIALIAS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert image colors
    img_array = 255 - img_array
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    # Reshape for model input
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

def predict_digit(image_path, model_path='mnist_model.h5'):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Prepare the image
    prepared_image = load_and_prepare_image(image_path)

    # Make a prediction
    prediction = model.predict(prepared_image)
    predicted_digit = np.argmax(prediction)
    
    # Display the image and prediction
    plt.imshow(prepared_image.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.show()

if __name__ == "__main__":
    # Specify the path to your JPG image and model
    image_path = 'num2.jpg'  # Update this path
    predict_digit(image_path)
