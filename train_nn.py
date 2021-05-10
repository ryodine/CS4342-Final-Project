import numpy as np
import tensorflow as tf

### Importing the model, which is defined in another file ###
import model_nn
model = model_nn.get_model()

### Importing the data, which is defined in another file ###
import data_nn
train_generator, validation_generator, _ = data_nn.get_data_generators()

############################################################
#
#   Model Training
#
############################################################

checkpoint_filepath = './checkpoint_model'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model_early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

model.fit(x=train_generator,
          validation_data=validation_generator,
          callbacks=[model_checkpoint_callback],
          epochs=300)
