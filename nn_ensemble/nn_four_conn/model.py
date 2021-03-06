import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Dropout
import os

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=7, input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.load_weights(os.path.dirname(os.path.abspath(__file__))  + '/checkpoint_model')

    return model

if __name__ == "__main__":
    mod = get_model()
    tf.keras.utils.plot_model(mod, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print("saved model image")