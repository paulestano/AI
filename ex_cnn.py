# Import useful modules from Keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical

import numpy as np

# Import the dataset MNIST
from keras.datasets import cifar10

# Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Show the data size
print (x_train.shape)
# The first dimension is the number of samples (6000).
# Each image is a matrix of 28 x 28 gray scale pixels.

# Rescale the images to vectors
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
print (x_train.shape)

x_train = x_train.astype(np.float32)
x_train /= 255.
x_train -= 0.5
x_train *= 2.
x_test = x_test.astype(np.float32)
x_test /= 255.
x_test -= 0.5
x_test *= 2.

# Convert labels (numbers) to categorical vectors
print (y_train.shape)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print (y_train.shape)

# Define our model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))

# model.add(Conv2D(64, (5, 5)))
# model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

# Compile the model with SGD
model.compile(optimizer='sgd', loss='categorical_crossentropy',
        metrics=['accuracy'])

model.summary()

from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='./')

# Training
model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=20,
        callbacks=[tensorboard])
