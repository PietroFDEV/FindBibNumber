import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load datasets
dataset1 = load_dataset("tanganke/emnist_letters")
dataset2 = load_dataset("pittawat/letter_recognition")

# Preprocess function for dataset
def preprocess_images(dataset, label_offset=0):
    images = []
    labels = []
    
    # Iterate over each item in the dataset's training split
    for item in dataset['train']:
        # Since the image is already a PIL Image object, just convert it to grayscale
        image = item['image'].convert('L')
        image_array = np.array(image)

        # Append to lists
        images.append(image_array)
        labels.append(item['label'] + label_offset)
    
    images = np.array(images, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    labels = np.array(labels, dtype=np.int32)

    # Reshape images to include the channel dimension (28, 28, 1)
    images = np.expand_dims(images, axis=-1)
    
    return images, labels

# Preprocess both datasets
images1, labels1 = preprocess_images(dataset1)
images2, labels2 = preprocess_images(dataset2, label_offset=26)  # Offset for different alphabet set

# Combine datasets
images = np.concatenate((images1, images2), axis=0)
labels = np.concatenate((labels1, labels2), axis=0)
labels = to_categorical(labels)  # Convert labels to one-hot encoding

# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(52, activation='softmax')  # 52 classes for A-Z, a-z
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(images, labels, epochs=10)

# Save the model
model.save('letter_classification_model.h5')
