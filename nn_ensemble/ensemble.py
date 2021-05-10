from nn_two_conn.model import get_model as get_model_2
from nn_four_conn.model import get_model as get_model_4
from nn_five_conn.model import get_model as get_model_5
from nn_five_conn_three_dense.model import get_model as get_model_5_1
from nn_six_conn.model import get_model as get_model_6
import numpy as np
import sys
sys.path.append("..")
from data_nn import get_data_generators
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression



train, validate, test = get_data_generators(csv="../train.csv", imdir="../train_images")

models = [get_model_2(), get_model_4(), get_model_5(), get_model_5_1(), get_model_6()]

def predict_raw_consensus(data_gen):
    predictions = [np.argmax(model.predict(data_gen), axis=1) for model in models]
    predmode = stats.mode(np.vstack(predictions))

    print(predictions)
    return predmode[0].flatten()

def train_ensemble(training_data):
    predictions = np.hstack([model.predict(training_data) for model in models])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(predictions, training_data.labels)
    return clf
    

def predict(generator, ensemble):
    ensemble_raw = np.hstack([model.predict(generator) for model in models])
    return ensemble.predict(ensemble_raw)

clf = train_ensemble(train)

y_pred = predict(test, clf)
print(y_pred)
y_true = test.labels

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
