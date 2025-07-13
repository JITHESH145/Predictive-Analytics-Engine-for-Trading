import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


X_train = joblib.load('X_train.pkl')
y_train = joblib.load('y_train.pkl')

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf')
}

for name, model in models.items():
    print(f'Training {name}...')
    model.fit(X_train, y_train)
    joblib.dump(model, f'{name.replace(' ', '_')}.pkl')
    print(f'{name} trained and saved.')

print('\nModels trained and saved successfully.')

from sklearn.model_selection import GridSearchCV

ridge_params = {'alpha': [0.1, 1.0, 10.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)
print("Best Ridge params:", ridge_grid.best_params_)
joblib.dump(ridge_grid.best_estimator_, "Tuned_Ridge.pkl")

lasso_params = {'alpha': [0.01, 0.1, 1.0]}
lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
print("Best Lasso params:", lasso_grid.best_params_)
joblib.dump(lasso_grid.best_estimator_, "Tuned_Lasso.pkl")

rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)
print("Best Random Forest params:", rf_grid.best_params_)
joblib.dump(rf_grid.best_estimator_, "Tuned_Random_Forest.pkl")
