import argparse
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Linear Support Vector Machine for given csv input')
parser.add_argument('trainInput', type=str)
parser.add_argument('testInput', type=str)
parser.add_argument('outputFile', type=str)

args = parser.parse_args()

print("Loading Data...")

df_train = pd.read_csv(args.trainInput)
X_train = df_train.drop('non-information', axis=1).values
Y_train = df_train[['non-information']].values

df_test = pd.read_csv(args.testInput)
X_test = df_test.drop('non-information', axis=1).values
Y_test = df_test[['non-information']].values


print("Data loaded")

bestAcc = 0

print("Training classifiers...")

model = RandomForestClassifier(n_estimators=150, max_depth=5)
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)

if np.round(accuracy, 4) > bestAcc:
    bestAcc = accuracy
    bestclf = model
    bestX_train = X_train
    bestX_test = X_test
    bestY_train = Y_train
    bestY_test = Y_test

y_pred_class = bestclf.predict(bestX_test)

bestY_test = bestY_test.reshape((bestY_test.size, ))
s = pd.Series(data=bestY_test)

print('\n')
print('Distribution of test labels')
print(s.value_counts())

print('\n')
print('Percentage of most common value: ')
print(s.value_counts().head(1) / len(s))

confusion = metrics.confusion_matrix(bestY_test, y_pred_class)

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print('\n')

print('TP: ' + str(TP))
print('TN: ' + str(TN))
print('FP: ' + str(FP))
print('FN: ' + str(FN))
print('\n')

print('Classification Accuracy:')
print(metrics.accuracy_score(bestY_test, y_pred_class))
print('\n')

print('Classification Error:')
print(1 - metrics.accuracy_score(bestY_test, y_pred_class))
print('\n')

print('Sensitivity: ')
print(metrics.recall_score(bestY_test, y_pred_class))
print('\n')

print('Specificity/Recall: ')
recall = TN / float(TN + FP)
print(str(recall))
print('\n')

print('False Positive Rate: ')
print(FP / float(TN + FP))
print('\n')

print('Precision: ')
precision = metrics.precision_score(bestY_test, y_pred_class)
print(str(precision))
print('\n')

print('F1 Score:')
f1 = 2/(pow(recall, -1) + pow(precision, -1))
print(str(f1))
joblib.dump(bestclf, args.outputFile)

print('MCC:')
mcc = (TP*TN-FP*FN)/np.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
print(str(mcc))
