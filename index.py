# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# get sample data from keras datasets online
data = keras.datasets.fashion_mnist

# destructure test images and labels from the load data function
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# specify class names for the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the layers in the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# set additional parameters for our model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)

# test the model
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# print('Tested Acc: ', test_acc)

# Use the model to predict
prediction = model.predict(test_images)
print(np.argmax(prediction[0]))
