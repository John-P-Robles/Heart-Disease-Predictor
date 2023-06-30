import pandas as pd
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('data_scaled.csv')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
param_grid = {
    'C': [15],
    'gamma': [2],
    'kernel': ['rbf'],
}
X = df.drop(['10-year_risk'], axis=1)
y = df['10-year_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()

SVM = SVC(max_iter=1000)
model = GridSearchCV(estimator=SVM, param_grid = param_grid,  param_grid, refit=True,cv=10, n_jobs=-1, verbose=3)
model.fit(X_train,y_train)

pickle.dump(model, open('model.pkl', 'wb'))







