from google.colab import drive
drive.mount('/content/drive')
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
import matplotlib.pyplot as plt

# Path to dataset
dataset_path = '/content/drive/MyDrive/knuckle'  # Adjust this path if needed

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The specified path {dataset_path} does not exist. Please check the path and try again.")

# Parameters
img_size = (128, 128)  # Resize images to 128x128
num_classes = 100  # Number of persons (classes)

# Data arrays
images = []
labels = []

# Load the dataset
for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)
    if os.path.isdir(person_path):
        label = int(person_folder.replace("Person", "")) - 1  # Label from 0 to 99
        for image_file in os.listdir(person_path):
            img_path = os.path.join(person_path, image_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)  # Resize image
                images.append(img)
                labels.append(label)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Check the shape of images and adjust data type if necessary
print(f"Shape of images: {images.shape}")
images = images.astype('float32')

# Normalize images
images = images / 255.0

# One-hot encode the labels
labels = to_categorical(labels, num_classes=num_classes)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Print the shape of the training data
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Define the input layer
input_layer = Input(shape=(128, 128, 3))

# Use a pre-trained ResNet50 model as the base, excluding the top fully-connected layers
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model to check shapes
model.summary()

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test), epochs=20)

# Save the model after training
model.save('/content/drive/MyDrive/knuckle_auth_model_resnet50_functional.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Plot accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model('/content/drive/MyDrive/knuckle_auth_model_resnet50_functional.h5')
import os import cv2
import numpy as np

# Function to preprocess the image with error handling
def preprocess_image(image_path):
    img_size = (128, 128)

    # Check if the file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file {image_path} does not exist.")

    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}. Please check the file format and path.")

    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
def authenticate_knuckle(image_path, true_label):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Predict the person (class) using the trained model
    predictions = model.predict(img)

    # Get the predicted class and probabilities
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probabilities = predictions[0]  # Get the probabilities for the classes

    # Display predicted class and probabilities
    print(f"Predicted Class: {predicted_class}, True Label: {true_label}")
    print("Prediction Probabilities:", predicted_probabilities)

    # Check if the predicted class matches the true label
    if predicted_class == true_label:
        print("Authentication Successful! Access Granted.")
    else:
        print("Authentication Failed! Access Denied.")
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.title('Training and Validation Accuracy and Loss')
plt.show()
# Evaluate on a few training samples
for i in range(5):  # Change this to evaluate more samples
    img = preprocess_image(f'/content/drive/MyDrive/knuckle/Person01/{i + 1}.bmp')
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    print(f"Predicted Class for Person01 Image {i + 1}: {predicted_class}, Probabilities: {predictions}")
# Example: Test with a knuckle image of Person01 (label should be 0)
authenticate_knuckle('/content/drive/MyDrive/knuckle/Person95/471.bmp', 0)
