# predict_multiple_digits.py
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # convert to grayscale
    img = np.array(img)
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)[1]  # invert and binarize
    return img

def extract_digits(img):
    # Find contours of the digits
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_imgs = []
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Extract the digit (extend margins a bit)
        extended = max(w, h) // 2
        digit = img[max(0, y-extended):y+h+extended, max(0, x-extended):x+w+extended]
        if digit.shape[0] == 0 or digit.shape[1] == 0:
            continue
        # Resize to 28x28
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit_imgs.append(digit)
    return digit_imgs

def predict_digits(image_path):
    model = tf.keras.models.load_model('mnist_model.h5')
    print("Model loaded successfully")
    
    processed_image = load_and_preprocess_image(image_path)
    digit_images = extract_digits(processed_image)
    
    full_number = ""
    plt.figure(figsize=(10, len(digit_images) * 3))
    for i, digit_img in enumerate(digit_images):
        digit_img = digit_img.reshape((1, 28, 28, 1)) / 255.0  # Normalize
        prediction = model.predict(digit_img)
        predicted_digit = np.argmax(prediction)
        full_number += str(predicted_digit)
        
        plt.subplot(len(digit_images), 1, i+1)
        plt.imshow(digit_img.reshape(28, 28), cmap='gray')
        plt.title(f'Predicted Digit: {predicted_digit}')
    plt.show()
    
    print("Full number detected:", full_number)

# Example usage
predict_digits('nums.jpg')  # replace 'path_to_your_image.jpg' with your image path
