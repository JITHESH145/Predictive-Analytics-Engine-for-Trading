import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')


model_files = [
    'Linear_Regression.pkl',
    'Polynomial_Regression.pkl',
    'Ridge.pkl',
    'Lasso.pkl',
    'Decision_Tree_Regressor.pkl',
    'Random_Forest_Regressor.pkl',
    'Support_Vector_Regression.pkl',
    'Tuned_Ridge.pkl',
    'Tuned_Lasso.pkl',
    'Tuned_Random_Forest.pkl'
]


results = {}

for file in model_files:
    try:
        name = file.replace('_', ' ').replace('.pkl', '')
        print(f"Evaluating {name}...")
        model = joblib.load(file)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R² Score': r2}
        
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label='Actual Next Close')
        plt.plot(y_pred, label='Predicted Next Close')
        plt.title(f'{name} - Predictions vs Actual')
        plt.xlabel('Test Samples')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()
    except FileNotFoundError:
        print(f"Could not find model file: {file}. Skipping.")


if results:
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by='RMSE')  
    print("\nModel Performance Comparison:")
    print(results_df)
else:
    print("No models were evaluated.")
