import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess_image(gray_img):
    # Thresholding to get binary image and inversion
    _, thresh = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)
    # Optionally add dilation here if characters are not well segmented
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(thresh, kernel, iterations=1)

def find_and_crop_characters(thresh_img):
    ctrs, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    for ctr in ctrs:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        if w > 5 and h > 15:  # Filter out too small contours that may not be characters
            roi = thresh_img[y:y+h, x:x+w]
            characters.append((roi, (x, y, w, h)))
    return characters

def predict_characters(characters, model):
    predictions = []
    for char, bbox in characters:
        # Resize to match the input shape of the model, e.g., 28x28
        char = cv2.resize(char, (28, 28))
        char = np.expand_dims(char, axis=-1)  # Add channel dimension
        char = np.expand_dims(char, axis=0)   # Add batch dimension
        char = char / 255.0  # Normalize pixel values
        pred = model.predict(char)
        predicted_digit = np.argmax(pred)
        predictions.append((predicted_digit, bbox))
    return predictions

# Main execution
image_path = 'nums.jpg'  # Specific path to the image
gray_image = load_image(image_path)
if gray_image is not None:
    thresh_image = preprocess_image(gray_image)
    characters = find_and_crop_characters(thresh_image)
    model = tf.keras.models.load_model('multi_digit_model.h5')  # Load your pre-trained model
    predictions = predict_characters(characters, model)

    # Visualizing results
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    for digit, (x, y, w, h) in predictions:
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=None, color='red'))
        plt.text(x, y - 10, str(digit), fontsize=12, color='red')
    plt.show()
