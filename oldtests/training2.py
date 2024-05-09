import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

def prepare_images(num_digits=2, num_samples=60000):
    (train_images, train_labels), _ = datasets.mnist.load_data()
    train_images = train_images / 255.0
    train_images = np.expand_dims(train_images, axis=-1)

    # Create new images with two digits side by side
    new_images = np.zeros((num_samples, 28, 28 * num_digits, 1))
    new_labels = []
    for i in range(num_samples):
        indices = np.random.choice(len(train_images), num_digits, replace=False)
        new_image = np.hstack([train_images[idx] for idx in indices])
        label = int(''.join(str(train_labels[idx]) for idx in indices))
        new_images[i] = new_image
        new_labels.append(label)
    
    return new_images, np.array(new_labels)

# Prepare synthetic images
train_images, train_labels = prepare_images()

# Define a simple CNN suitable for wide images with multiple digits
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 56, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(100, activation='softmax')  # Output layer for 0-99 if 2 digits
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=64)

# Save the model
model.save('multi_digit_model.h5')
print("Model saved to multi_digit_model.h5")
