from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, balanced_accuracy_score
import xgboost as xgb
import csv
import pandas as pd

# load data
dataset = 'text or csv file'
target = 'text or csv file'


with open(dataset, 'r') as textFile:
     params1t = textFile.readlines()
     meta = [[num for num in s.replace('[', '').replace(']', '').replace('\n', '0.0').replace('[[', '').replace(']]', '').split(',')] for s in params1t]

X = np.array(meta)

#X = pd.read_csv(dataset)
y = pd.read_csv(target)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.10)


model = XGBClassifier(eta=0.3,max_depth=6,objective='multi:softmax',num_class=7,scale_pos_weight=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

pred = [round(value) for value in y_pred]

# evaluate predictions

best_preds = np.asarray([np.argmax(line) for line in pred])

print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("Balanced Accuracy = {}".format(balanced_accuracy_score(y_test, pred)))

cm = confusion_matrix(y_test, pred)
print("Confusion matrix = {}".format(cm))

# feature importance

# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()
pyplot.savefig('save .jpg image')
# plot feature importance
plot_importance(model)
pyplot.show()
pyplot.savefig('save .jpg image')