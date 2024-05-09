import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize and reshape the data as before
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

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
    processed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    plt.imshow(processed_image.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.show()

# Test the prediction function with a new image
predict_digit('num2.jpg')  # replace 'path_to_your_image.jpg' with your image path
