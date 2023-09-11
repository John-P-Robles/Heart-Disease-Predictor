import pandas as pd
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('data_scaled.csv')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
param_grid = {
    'max_depth': [10,20,30,40,50],
    'n_estimators': [90,100,110,120,130],
    'min_samples_split': [6,8,10,12,14],
}
X = df.drop(['10-year_risk'], axis=1)
y = df['10-year_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()

RFC = RandomForestClassifier()
model = GridSearchCV(estimator=RFC, param_grid = param_grid,cv=15, n_jobs=-1, verbose=3)
model.fit(X_train,y_train)

pickle.dump(model, open('model.pkl', 'wb'))







