import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, models, layers
import cv2 as cv

# Load the CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize image data to 0–1 range
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names
class_name = ['Planes', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# # Display the first 16 images and their labels
# plt.figure(figsize=(8, 8))
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.yticks([])
#     plt.xticks([])
#     plt.imshow(training_images[i])
#     plt.xlabel(class_name[training_labels[i][0]])
# plt.tight_layout()
# plt.show()

# Use a subset of the dataset
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:40000]
testing_labels = testing_labels[:40000]

# # Build the CNN model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(training_images, training_labels, epochs=10,
#           validation_data=(testing_images, testing_labels))

# # Evaluate the model
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}, Accuracy: {accuracy}")

# # Save the model
# model.save('image_classifier.keras')  # recommended

model = models.load_model('image_classifier.keras')  # Load the saved model

# Load and preprocess the image
img = cv.imread(r"C:\Users\Mawuenyefia Hunorkpa\Desktop\PYTHON PROJECTS\Image_Classification\car.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))  # resize to match model input
img = img / 255.0  # normalize

# Display the image
plt.imshow(img, cmap=plt.cm.binary)
plt.axis('off')
plt.show()

# Predict
prediction = model.predict(np.array([img]))  # input must be batch-shaped
index = np.argmax(prediction)
print(f"Prediction is: {class_name[index]}")
