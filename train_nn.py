import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Dropout, LayerNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

##################
# Threading
tf.config.threading.set_inter_op_parallelism_threads(10)

############################################################
#
#   Load Image Data
#
############################################################

df = pd.read_csv("train.csv").astype(str)

splits = np.split(df, [6*len(df)//10, 8*len(df)//10])

batch_size = 32

# Training set will have some data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True)

# Create a second image data generator that onloy does the resizing, not augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    splits[0],
    'train_images',
    x_col='image_id',
    y_col='label',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = train_datagen.flow_from_dataframe(
    splits[1],
    'train_images',
    x_col='image_id',
    y_col='label',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical')


############################################################
#
#   Model Definition
#
############################################################

#create model
model = Sequential()
#add model layer

"""
model.add(Conv2D(50, kernel_size=15, input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(LayerNormalization())
model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(Conv2D(255, (3, 3)))
model.add(Activation('relu'))
model.add(LayerNormalization())
model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))
"""

"""
model.add(Conv2D(32, kernel_size=7, input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
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
"""


model.add(Conv2D(32, kernel_size=15, input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(Conv2D(32, kernel_size=7))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))


# the model so far outputs 3D feature maps (height, width, features)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


############################################################
#
#   Model
#
############################################################

checkpoint_filepath = './checkpoint_model'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

"""
model_early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
"""

model.fit(x=train_generator,
          validation_data=validation_generator,
          callbacks=[model_checkpoint_callback],
          epochs=150)