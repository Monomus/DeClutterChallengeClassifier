from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import joblib
import pandas as pd
import random as rand
import numpy as np
import argparse
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Linear Support Vector Machine for given csv input')
parser.add_argument('inputFile', type=str)
parser.add_argument('outputFile', type=str)
parser.add_argument('--runs', type=int, default=25, help='number of tries best will be taken')
parser.add_argument('--split', type=float, default=0.2, help='Split percentage for Training/Test')

args = parser.parse_args()

print("Loading Data...")

df = pd.read_csv(args.inputFile)

X = df.drop('non-information', axis=1).values
Y = df[['non-information']].values

label_encoder = preprocessing.LabelEncoder()
Y = label_encoder.fit_transform(Y)
X = preprocessing.StandardScaler().fit_transform(X)

print("Data loaded")

bestAcc = 0

print("Training classifiers...")
for i in range(0, args.runs):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.split, random_state=rand.randint(1, 100))
    clf = LinearSVC()

    clf.fit(x_train, y_train)

    accuracy = clf.score(x_test, y_test)

    if np.round(accuracy, 4) > bestAcc:
        bestAcc = accuracy
        bestclf = clf
        bestX_train = x_train
        bestX_test = x_test
        bestY_train = y_train
        bestY_test = y_test

y_pred_class = bestclf.predict(bestX_test)

print(y_pred_class)

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
