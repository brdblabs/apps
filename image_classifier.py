import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def main():
  st.title('Cifar10 Web Classifier')
  st.write("Upload any image that you think fits into one of the classes and see if the prediction is correct.")

  file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
  if file:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    resized_image = image.resize((32, 32))
    img_array = np.array(resized_image) / 255
    img_array = img_array.reshape((1, 32, 32, 3))

    # Progress bar while model is running
    st.write('Please standby while your image is analyzed...')

    # Preprocessing
    (X_train, y_train), (X_val, y_val) = cifar10.load_data()
    X_train = X_train / 255
    X_val = X_val /255

    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)

    # Create model
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(1000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_val, y_val))

    predictions = model.predict(img_array)
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'dog', 'deer', 'frog', 'horse', 'ship', 'truck']

    fig, ax = plt.subplots()
    y_pos = np.arange(len(cifar10_classes))
    ax.barh(y_pos, predictions[0], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cifar10_classes)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title('CIFAR10 Predictions')

    st.pyplot(fig)

  else:
    st.text('You have not uploaded an image yet.')

if __name__ == '__main__':
  main()
