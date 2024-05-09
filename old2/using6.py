import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def load_and_predict():
    # Load the model
    model = tf.keras.models.load_model('mnist_model.h5')
    print("Model loaded successfully.")

    # Load MNIST data for testing
    _, (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images = test_images / 255.0
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    # Select a few images to predict
    test_samples = test_images[:10]
    predictions = model.predict(test_samples)

    # Plot the results
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_samples[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(f"Actual: {test_labels[i]} - Predicted: {np.argmax(predictions[i])}")
    plt.show()

if __name__ == "__main__":
    load_and_predict()
