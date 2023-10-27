import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

# Preprocessing
(X_train, y_train), (X_val, y_val) = cifar10.load_data()
X_train = X_train / 255
X_val = X_val /255

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# Create the model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", kernel_initializer="he_uniform", padding="same", input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu", kernel_initializer="he_uniform", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),
    Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"),
    BatchNormalization(),
    Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),
    Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"),
    BatchNormalization(),
    Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation="relu", kernel_initializer="he_uniform"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, rotation_range=20)
train_aug = datagen.flow(X_train, y_train)
steps = int(X_train.shape[0] / 64)

# Fit the model and save
history = model.fit_generator(train_aug, epochs=200, steps_per_epoch=steps, validation_data=(X_val, y_val))
model.save('cifar10_model.h5')
