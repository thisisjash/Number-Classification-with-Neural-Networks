import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from tensorflow import keras
import tensorflow as tf
# Imported Necessary Modules

# Loading the Fashion Dataset into Dataframe
data = keras.datasets.fashion_mnist

# Splitting the Data, into Testing Set and Training Set
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# we will define a list of the class names and pre-process images. We do this
# by dividing each image by 255. Since each image is greyscale we are
# simply scaling the pixel values down to make computations easier for
# our model.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = test_images/255.0
test_images = test_images/255.0

# Creating our Neural Network Model by using the Sequential Object from
# Keras. A Sequential model simply defines a sequence of layers starting with
# the input layer and ending with the output layer. Our model will have
# 3 layers, and input layer of 784 neurons (representing all of the
# 28x28 pixels in a picture) a hidden layer of an arbitrary 128 neurons
# and an output layer of 10 neurons representing the probability of the picture
# being each of the 10 classes.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# The role of the Flatten layer in Keras is super simple:
# A flatten operation on a tensor reshapes the tensor to have
# the shape that is equal to the number of elements contained
# in tensor non including the batch dimension. The Data here
# is actually 28X28. Flatten Functions makes it 784(28*28) inputs

# Dense Method creates regular densely connected NN Layer
# Activation Method, describes the way and sequence to activate
# the neural nodes

# There are other arguments for Dense Function that characterise
# NN like initializer, regularizer, constraint of kernel, use_bias

# Configures the model for training.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# optimizer, loss are must for Configuring the model.
# There are various optimizer techniques like adam, SGD, RMSprop etc
# Loss is for error Computation. We can use different loss for different
# outputs by passing dictionary of different losses.
# List of metrics to be evaluated by the model during training and
# testing. Typically you will use metrics=['accuracy']. To specify
# different metrics for different outputs of a multi-output model,
# you could also pass a dictionary

train_labels = np.reshape(train_labels, 6)

# Feed the data to model
model.fit(train_images, train_labels, epochs=5)

# epochs is "How Many Times we shall train each dataset"

# Testing the data with the Test Data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Printing accuracy
print('\nTest accuracy:', test_acc)
