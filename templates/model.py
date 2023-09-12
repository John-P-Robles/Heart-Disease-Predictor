from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('norm_CHD.csv')

param_grid = {
    'C': [5,10,15],
    'gamma': [1,2,3],
    'kernel': ['rbf'],
    'class_weight':['balanced', None],
}
X = df.drop(['10-year_risk','Unnamed: 0'], axis=1)
y = df['10-year_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

SVM2 = SVC(gamma='scale')
model = GridSearchCV(estimator=SVM2, param_grid = param_grid, refit=True,cv=10, n_jobs=-1, verbose=3)
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))

