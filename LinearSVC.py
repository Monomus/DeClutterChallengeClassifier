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

bestmcc = 0
print("Training classifiers...")
for i in range(0, args.runs):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.split, random_state=rand.randint(1, 100))
    clf = LinearSVC()

    clf.fit(x_train, y_train)

    y_pred_class = clf.predict(x_test)

    confusion = metrics.confusion_matrix(y_test, y_pred_class)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    recall = TN / float(TN + FP)

    acc = metrics.accuracy_score(y_test, y_pred_class)

    precision = metrics.precision_score(y_test, y_pred_class)

    f1 = 2 / (pow(recall, -1) + pow(precision, -1))

    mcc = (TP * TN - FP * FN) / np.sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

    combinedMetric = (f1 + mcc) / 2

    if np.round(mcc, 4) > bestmcc:
        bestclf = clf
        bestAcc = acc
        bestf1 = f1
        bestmcc = mcc

print('\n')
print('Classification Accuracy:')
print(str(bestAcc))
print('F1 Score:')
print(str(bestf1))
print('MCC:')
print(str(bestmcc))

joblib.dump(bestclf, args.outputFile)