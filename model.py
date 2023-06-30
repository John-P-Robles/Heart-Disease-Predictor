import pandas as pd

df = pd.read_csv('data_scaled.csv')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import pickle
X = df.drop(['10-year_risk'], axis=1)
y = df['10-year_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()


model = SVC(max_iter=1000)
model.fit(X_train,y_train)

pickle.dump(SVM, open('model.pkl', 'wb'))







