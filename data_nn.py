from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

############################################################
#
#   Load Image Data
#
############################################################

df = pd.read_csv("train.csv").astype(str)

splits = np.split(df, [6*len(df)//10, 8*len(df)//10])

batch_size = 16

# Training set will have some data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True)

# Create a second image data generator that onloy does the resizing, not augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

def get_data_generators():

    train_generator = train_datagen.flow_from_dataframe(
        splits[0],
        'train_images',
        x_col='image_id',
        y_col='label',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_dataframe(
        splits[1],
        'train_images',
        x_col='image_id',
        y_col='label',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_dataframe(
        splits[2],
        'train_images',
        x_col='image_id',
        y_col='label',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator, test_generator