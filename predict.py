import joblib
import pandas as pd
from sklearn.svm import LinearSVC

svc = joblib.load('treeNew.pkl')

df_test = pd.read_csv('test_results.csv')
df_test_id = df_test
df_test = df_test.drop('ID', axis=1)
X_test = df_test.drop('non-information', axis=1).values
Y_test = df_test[['non-information']].values


predictedValues = svc.predict(X_test)

print(df_test_id)

df_test_id = df_test_id.filter(['ID'])

print(predictedValues)
print(df_test_id)

df_test_id.insert(1,"Predicted", predictedValues, True)

print(df_test_id)

df_test_id['Predicted'] = df_test_id['Predicted'].replace(True, 'yes')
df_test_id['Predicted'] = df_test_id['Predicted'].replace(False, 'no')


print(df_test_id)
df_test_id.to_csv('submission.csv', index=False)


