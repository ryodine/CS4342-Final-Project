import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

### Importing the model, which is defined in another file ###
import model_nn
model = model_nn.get_model()

### Importing the data, which is defined in another file ###
import data_nn
train_generator, validation_generator, test_generator = data_nn.get_data_generators()

checkpoint_filepath = './checkpoint_model'
model.load_weights(checkpoint_filepath)

preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)

labels = test_generator.labels
y_true = labels

print("Accuracy:", accuracy_score(y_true, y_pred))
print('Confusion Matrix')
print(confusion_matrix(y_true, y_pred))
print('Classification Report')
target_names = [
  "Cassava Bacterial Blight (CBB)",
  "Cassava Brown Streak Disease (CBSD)",
  "Cassava Green Mottle (CGM)",
  "Cassava Mosaic Disease (CMD)",
  "Healthy"
]
print(classification_report(y_true, y_pred, target_names=target_names))